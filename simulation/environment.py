"""
MuJoCo Simulation Environment for Franka Panda Robot.

Manages:
- Arena XML loading and physics stepping
- Franka Panda robot controller binding
- Workstation, target object, and goal monitoring
- Contact and collision detection
- Offscreen rendering and camera management
"""

import os
import yaml
import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple, Any

from simulation.robot import FrankaPandaRobot
from simulation.obstacles import ObstacleManager
from observation.observation_manager import ObservationManager
from safety.safety_monitor import SafetyMonitor
from safety.safety_supervisor import SafetySupervisor, SupervisorState


class SawmEnvironment:
    """Primary simulation environment for SAWM research prototype."""

    def __init__(self, config_path: str = "configs/simulation.yaml", safety_config_path: str = "configs/safety.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Load safety configuration if present
        self.safety_config_path = safety_config_path
        self.safety_config = self._load_config(safety_config_path) if os.path.exists(safety_config_path) else {}

        # Resolve model path
        model_path = self.config["scene"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MuJoCo model not found at: {model_path}")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Apply physics options
        self.frame_skip = int(self.config["scene"].get("frame_skip", 5))
        self.dt = float(self.model.opt.timestep)

        # Initialize robot controller
        self.robot = FrankaPandaRobot(self.model, self.data, self.config.get("robot", {}))

        # Initialize hazard & safety managers
        self.obstacle_manager = ObstacleManager(self.model, self.data, self.safety_config)
        self.observation_manager = ObservationManager(self.model, self.data, self.safety_config)
        self.safety_monitor = SafetyMonitor(self.safety_config)
        self.restricted_zones_enabled = True

        # Body IDs for monitoring
        self.table_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["environment"]["table_body_name"]
        )
        self.target_object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["environment"]["target_object_body_name"]
        )
        self.goal_zone_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.config["environment"]["goal_zone_body_name"]
        )

        # Default robot configurations
        self.home_qpos = np.array(self.config["robot"]["home_qpos"], dtype=np.float64)
        self.ready_qpos = np.array(self.config["robot"]["ready_qpos"], dtype=np.float64)
        self.approach_qpos = np.array(self.config["robot"]["approach_qpos"], dtype=np.float64)

        # Offscreen renderer cache
        self._renderer: Optional[mujoco.Renderer] = None
        self._render_width = int(self.config["rendering"].get("width", 1280))
        self._render_height = int(self.config["rendering"].get("height", 720))

        # Simulation-level recovery authorization tracking (not a security boundary)
        self._inhibiting_supervisor: Optional[SafetySupervisor] = None

        # Reset to initial state
        self.reset()

    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Lifecycle & Stepping
    # -------------------------------------------------------------------------
    def inhibit_arm_actuators(self, supervisor: Optional[SafetySupervisor] = None) -> None:
        """
        Genuinely inhibit arm actuators (Safe Torque Off) and bind the inhibition event
        to the specific SafetySupervisor instance responsible.

        If supervisor is None, actuators remain inhibited with zero torque;
        automatic substitution of a different supervisor is strictly prohibited.

        NOTE: Simulation-level recovery authorization mechanism, not a security boundary
        against arbitrary Python code execution.
        """
        self.robot.inhibit_arm_actuators()
        if self._inhibiting_supervisor is None and supervisor is not None:
            if not isinstance(supervisor, SafetySupervisor):
                raise TypeError(f"supervisor must be a genuine SafetySupervisor instance, got {type(supervisor)}")
            self._inhibiting_supervisor = supervisor

    def authorize_actuator_recovery(self, supervisor: Any) -> Tuple[bool, str]:
        """
        Explicit, validated actuator recovery authorization from the bound SafetySupervisor.

        CRITICAL RECOVERY INTEGRITY:
        Actuator restoration is authorized ONLY if:
        1. A genuine SafetySupervisor instance is provided (isinstance check).
        2. If arm actuators are inhibited, the supervisor instance matches the specific
           SafetySupervisor that initiated the active inhibition latch (self._inhibiting_supervisor).
        3. The supervisor is not latched (supervisor.is_latched is False).
        4. The supervisor provides verified evidence of completing an explicit operator-confirmed
           recovery handshake (supervisor.has_completed_recovery_handshake is True).
        5. The supervisor is currently in RECOVERY or NORMAL state. A supervisor in NORMAL cannot
           authorize recovery unless it is proven to have legitimately passed through RECOVERY.

        NOTE: Simulation-level recovery authorization mechanism, not a security boundary
        against arbitrary Python code execution.

        Returns:
            (authorized: bool, message: str)
        """
        if supervisor is None:
            return False, "RECOVERY_REJECTED: Valid SafetySupervisor instance required"

        if not isinstance(supervisor, SafetySupervisor):
            return False, "RECOVERY_REJECTED: Arbitrary or fake supervisor object rejected. Genuine SafetySupervisor instance required."

        if self.robot.arm_actuators_inhibited:
            if self._inhibiting_supervisor is None:
                return False, (
                    "RECOVERY_REJECTED: Active inhibition has no bound supervisor recorded. "
                    "Actuators remain inhibited; automatic substitution of an unauthorized supervisor "
                    "is prohibited."
                )

            if supervisor is not self._inhibiting_supervisor:
                return False, "RECOVERY_REJECTED: Supervisor identity mismatch. Recovery must be authorized by the specific SafetySupervisor instance that triggered inhibition."

        if supervisor.is_latched:
            return False, "RECOVERY_REJECTED: Supervisor is latched in emergency stop. Operator reset required."

        if not getattr(supervisor, "has_completed_recovery_handshake", False):
            return False, "RECOVERY_REJECTED: Supervisor has not completed an explicit operator-confirmed recovery handshake."

        if supervisor.state not in (SupervisorState.RECOVERY, SupervisorState.NORMAL):
            return False, f"RECOVERY_REJECTED: Supervisor must be in RECOVERY or NORMAL state, currently {supervisor.state}"

        self.robot._restore_arm_actuators()
        self._inhibiting_supervisor = None
        return True, f"RECOVERY_AUTHORIZED: Arm actuators re-enabled under supervisor state {supervisor.state.value}"

    def reset(
        self,
        initial_qpos: Optional[np.ndarray] = None,
        recovery_supervisor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Reset simulation to initial state.

        CRITICAL LATCH INTEGRITY:
        Does NOT unconditionally re-enable arm actuators if they were inhibited.
        An existing arm actuator inhibition latch persists across reset unless
        explicitly authorized via recovery_supervisor or authorize_actuator_recovery().

        Args:
            initial_qpos: Optional 7-dim arm joint configuration. Defaults to home_qpos.
            recovery_supervisor: Optional SafetySupervisor instance. If provided and authorized,
                                 restores arm actuators if they were previously inhibited.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Actuator restoration requires explicit validated recovery authorization
        if recovery_supervisor is not None:
            self.authorize_actuator_recovery(recovery_supervisor)

        # Reset safety monitor
        self.safety_monitor.reset()

        qpos = self.home_qpos if initial_qpos is None else initial_qpos
        self.robot.set_joint_positions_direct(qpos)
        self.robot.set_gripper_target(0.0)  # Open gripper

        # Forward kinematics and dynamics computation
        mujoco.mj_forward(self.model, self.data)

        # Step a few times to let contacts settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    def step(
        self,
        arm_targets: Optional[np.ndarray] = None,
        gripper_target: Optional[float] = None,
        actuator_inhibit: bool = False,
        supervisor: Optional[SafetySupervisor] = None,
    ) -> Dict[str, Any]:
        """
        Step simulation forward by frame_skip physics timesteps with commanded targets.

        Args:
            arm_targets: (7,) commanded arm joint positions in radians.
            gripper_target: commanded gripper actuator position (0=open, 255=closed).
            actuator_inhibit: if True, genuinely inhibits arm actuators (Safe Torque Off).
            supervisor: Optional SafetySupervisor instance responsible for inhibition event.

        Returns:
            Observation dictionary.
        """
        if actuator_inhibit:
            self.inhibit_arm_actuators(supervisor=supervisor)

        if not self.robot.arm_actuators_inhibited and arm_targets is not None:
            self.robot.set_joint_targets(arm_targets)

        if gripper_target is not None:
            self.robot.set_gripper_target(gripper_target)

        # Advance dynamic hazard proxy along trajectory
        self.obstacle_manager.step_proxy_trajectory()

        # Substep physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation()

    # -------------------------------------------------------------------------
    # Observation & State
    # -------------------------------------------------------------------------
    def get_target_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return target object position (3,) and orientation quaternion (4,)."""
        pos = np.array(self.data.xpos[self.target_object_body_id], dtype=np.float64)
        quat = np.array(self.data.xquat[self.target_object_body_id], dtype=np.float64)
        return pos, quat

    def get_goal_position(self) -> np.ndarray:
        """Return goal zone position (3,)."""
        return np.array(self.data.xpos[self.goal_zone_body_id], dtype=np.float64)

    def get_observation(self) -> Dict[str, Any]:
        """Collect complete state observation including kinematics, contacts, and safety telemetry."""
        obj_pos, obj_quat = self.get_target_object_pose()
        goal_pos = self.get_goal_position()
        ee_pos, ee_quat = self.robot.get_end_effector_pose()

        # Distance from end-effector to target object
        ee_to_obj_dist = float(np.linalg.norm(ee_pos - obj_pos))
        # Distance from object to goal
        obj_to_goal_dist = float(np.linalg.norm(obj_pos - goal_pos))

        # Milestone 2 Safety Computations
        distances = self.observation_manager.compute_distances(self.obstacle_manager)
        contacts = self.observation_manager.collect_contacts()
        zone_states = self.observation_manager.evaluate_restricted_zones(self.restricted_zones_enabled)
        safety_telemetry = self.safety_monitor.update(distances, contacts, zone_states)

        return {
            "time": float(self.data.time),
            "robot": self.robot.get_full_state(),
            "target_object": {
                "position": obj_pos,
                "quaternion": obj_quat,
                "dist_to_ee": ee_to_obj_dist,
            },
            "goal": {
                "position": goal_pos,
                "dist_to_object": obj_to_goal_dist,
            },
            "contacts": self.get_contacts_summary(),
            "contact_metrics": contacts,
            "joint_limits": self.robot.check_joint_limits(),
            # Milestone 2 safety additions
            "safety": safety_telemetry,
            "distances": distances,
            "zone_states": zone_states,
            "hazards": {
                "static_active": self.obstacle_manager.is_static_active(),
                "static_pos": self.obstacle_manager.get_static_obstacle_position(),
                "proxy_active": self.obstacle_manager.is_proxy_active(),
                "proxy_pos": self.obstacle_manager.get_proxy_position(),
            },
        }

    # -------------------------------------------------------------------------
    # Collision & Contact Detection
    # -------------------------------------------------------------------------
    def get_contacts(self) -> List[Dict[str, Any]]:
        """
        Inspect and return all active contact pairs in the simulation.
        
        Returns:
            List of contact records with geom/body names, contact point, and penetration depth.
        """
        contacts = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1_id = con.geom1
            geom2_id = con.geom2

            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id) or f"geom_{geom1_id}"
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id) or f"geom_{geom2_id}"

            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or f"body_{body1_id}"
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or f"body_{body2_id}"

            contact_info = {
                "geom1": geom1_name,
                "geom2": geom2_name,
                "body1": body1_name,
                "body2": body2_name,
                "pos": np.array(con.pos, dtype=np.float64),
                "dist": float(con.dist),  # negative indicates penetration
            }
            contacts.append(contact_info)
        return contacts

    def get_contacts_summary(self) -> Dict[str, Any]:
        """Summarize active contacts into high-level categories."""
        raw_contacts = self.get_contacts()
        robot_body_names = set([f"link{i}" for i in range(8)] + ["hand", "left_finger", "right_finger"])

        robot_table_contacts = []
        robot_object_contacts = []
        object_table_contacts = []
        other_contacts = []

        for c in raw_contacts:
            b1, b2 = c["body1"], c["body2"]
            is_b1_robot = b1 in robot_body_names
            is_b2_robot = b2 in robot_body_names
            is_b1_table = "table" in b1
            is_b2_table = "table" in b2
            is_b1_obj = "target_object" in b1
            is_b2_obj = "target_object" in b2

            if (is_b1_robot and is_b2_table) or (is_b2_robot and is_b1_table):
                robot_table_contacts.append(c)
            elif (is_b1_robot and is_b2_obj) or (is_b2_robot and is_b1_obj):
                robot_object_contacts.append(c)
            elif (is_b1_obj and is_b2_table) or (is_b2_obj and is_b1_table):
                object_table_contacts.append(c)
            else:
                other_contacts.append(c)

        robot_obstacle_contacts = [
            c for c in other_contacts
            if ((c["body1"] in robot_body_names and "static_obstacle" in c["body2"]) or
                (c["body2"] in robot_body_names and "static_obstacle" in c["body1"]))
        ]
        robot_proxy_contacts = [
            c for c in other_contacts
            if ((c["body1"] in robot_body_names and "human_proxy" in c["body2"]) or
                (c["body2"] in robot_body_names and "human_proxy" in c["body1"]))
        ]

        return {
            "total_contacts": len(raw_contacts),
            "robot_table_collision": len(robot_table_contacts) > 0,
            "robot_obstacle_collision": len(robot_obstacle_contacts) > 0,
            "robot_proxy_collision": len(robot_proxy_contacts) > 0,
            "robot_object_contact": len(robot_object_contacts) > 0,
            "object_table_contact": len(object_table_contacts) > 0,
            "raw_contacts": raw_contacts,
        }

    # -------------------------------------------------------------------------
    # Offscreen Rendering
    # -------------------------------------------------------------------------
    def render(self, camera: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """
        Render the scene to an RGB numpy array (H, W, 3) uint8.
        
        Args:
            camera: Camera name from model (e.g. 'workspace_cam', 'front_cam', 'top_cam').
            width: Image width in pixels.
            height: Image height in pixels.
        """
        w = width or self._render_width
        h = height or self._render_height
        cam_name = camera or self.config["rendering"].get("default_camera", "workspace_cam")

        if self._renderer is None or self._renderer.width != w or self._renderer.height != h:
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render()

    def close(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
