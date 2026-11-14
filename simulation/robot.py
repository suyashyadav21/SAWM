"""
Franka Panda Robot Interface for MuJoCo.

Encapsulates:
- Joint state retrieval (positions, velocities)
- Joint control via position actuators
- Gripper actuation (split tendon actuator)
- End-effector pose tracking (hand site / body)
- Joint limit monitoring and validation
- Smooth joint trajectory interpolation
"""

import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional


class FrankaPandaRobot:
    """Controller and state interface for Franka Emika Panda in MuJoCo."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[dict] = None):
        self.model = model
        self.data = data
        self.config = config or {}

        # Arm joint and actuator metadata
        self.joint_names = [f"joint{i+1}" for i in range(7)]
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.qpos_indices = [model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.qvel_indices = [model.jnt_dofadr[jid] for jid in self.joint_ids]

        self.actuator_names = [f"actuator{i+1}" for i in range(7)]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names]

        # Gripper metadata
        self.gripper_actuator_name = "actuator8"
        self.gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_actuator_name)

        # Hand (end-effector) body
        self.hand_body_name = "hand"
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.hand_body_name)

        # Finger joint IDs
        self.finger_joint_names = ["finger_joint1", "finger_joint2"]
        self.finger_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.finger_joint_names]
        self.finger_qpos_indices = [model.jnt_qposadr[jid] for jid in self.finger_joint_ids]

        # Joint position limits from model
        self.joint_limits = {}
        for i, jid in enumerate(self.joint_ids):
            limit_range = model.jnt_range[jid]
            self.joint_limits[self.joint_names[i]] = (float(limit_range[0]), float(limit_range[1]))

        # Gripper limit
        self.gripper_ctrl_range = (
            float(model.actuator_ctrlrange[self.gripper_actuator_id][0]),
            float(model.actuator_ctrlrange[self.gripper_actuator_id][1])
        )

    # -------------------------------------------------------------------------
    # State Inspection
    # -------------------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        """Return current arm joint positions in radians (7,)."""
        return np.array([self.data.qpos[idx] for idx in self.qpos_indices], dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        """Return current arm joint velocities in rad/s (7,)."""
        return np.array([self.data.qvel[idx] for idx in self.qvel_indices], dtype=np.float64)

    def get_gripper_position(self) -> float:
        """Return gripper finger opening distance in meters (0 to ~0.04m)."""
        return float(self.data.qpos[self.finger_qpos_indices[0]])

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return end-effector position (3,) in world frame and orientation quaternion (4,) [w, x, y, z].
        """
        pos = np.array(self.data.xpos[self.hand_body_id], dtype=np.float64)
        quat = np.array(self.data.xquat[self.hand_body_id], dtype=np.float64)
        return pos, quat

    def get_full_state(self) -> Dict[str, np.ndarray]:
        """Return structured dictionary of the robot state."""
        ee_pos, ee_quat = self.get_end_effector_pose()
        return {
            "joint_positions": self.get_joint_positions(),
            "joint_velocities": self.get_joint_velocities(),
            "ee_position": ee_pos,
            "ee_quaternion": ee_quat,
            "gripper_position": self.get_gripper_position(),
        }

    # -------------------------------------------------------------------------
    # Control Interface
    # -------------------------------------------------------------------------
    def set_joint_targets(self, targets: np.ndarray, clip_to_limits: bool = True) -> None:
        """
        Set target positions for the 7 arm actuators.
        
        Args:
            targets: Array of 7 target angles in radians.
            clip_to_limits: If True, clips commanded targets within joint limits.
        """
        assert len(targets) == 7, f"Expected 7 joint targets, got {len(targets)}"
        for i, act_id in enumerate(self.actuator_ids):
            val = float(targets[i])
            if clip_to_limits:
                jname = self.joint_names[i]
                low, high = self.joint_limits[jname]
                val = float(np.clip(val, low, high))
            self.data.ctrl[act_id] = val

    def set_gripper_target(self, target: float) -> None:
        """
        Set target for gripper actuator.
        
        Args:
            target: Value between 0.0 (open) and 255.0 (closed).
        """
        clamped = float(np.clip(target, self.gripper_ctrl_range[0], self.gripper_ctrl_range[1]))
        self.data.ctrl[self.gripper_actuator_id] = clamped

    def set_joint_positions_direct(self, qpos: np.ndarray) -> None:
        """Directly set joint positions in qpos (kinematic teleport, resets velocities)."""
        assert len(qpos) == 7, f"Expected 7 joint positions, got {len(qpos)}"
        for i, idx in enumerate(self.qpos_indices):
            self.data.qpos[idx] = qpos[i]
            self.data.qvel[self.qvel_indices[i]] = 0.0
        # Also set ctrl to match
        self.set_joint_targets(qpos)

    # -------------------------------------------------------------------------
    # Safety & Limit Verification
    # -------------------------------------------------------------------------
    def check_joint_limits(self, margin_rad: float = 0.05) -> Dict[str, dict]:
        """
        Check if joints are within safe operational limits.
        
        Args:
            margin_rad: Warning boundary margin in radians.
            
        Returns:
            Dictionary with per-joint status:
            {
                "joint1": {
                    "position": float,
                    "limits": (low, high),
                    "violated": bool,
                    "in_margin": bool,
                    "margin_remaining": float
                },
                ...
            }
        """
        status = {}
        q = self.get_joint_positions()
        for i, name in enumerate(self.joint_names):
            val = q[i]
            low, high = self.joint_limits[name]
            violated = bool(val < low or val > high)
            dist_to_low = val - low
            dist_to_high = high - val
            min_dist = min(dist_to_low, dist_to_high)
            in_margin = bool(min_dist < margin_rad)

            status[name] = {
                "position": float(val),
                "limits": (low, high),
                "violated": violated,
                "in_margin": in_margin,
                "margin_remaining": float(min_dist),
            }
        return status

    def is_safe_configuration(self) -> bool:
        """Returns True if all joints are strictly within their hardware limits."""
        checks = self.check_joint_limits(margin_rad=0.0)
        return not any(j["violated"] for j in checks.values())

    # -------------------------------------------------------------------------
    # Trajectory Generation
    # -------------------------------------------------------------------------
    @staticmethod
    def interpolate_trajectory(
        start_qpos: np.ndarray,
        target_qpos: np.ndarray,
        steps: int,
        method: str = "cosine"
    ) -> List[np.ndarray]:
        """
        Generate smooth trajectory from start_qpos to target_qpos.
        
        Args:
            start_qpos: Initial joint configuration (7,).
            target_qpos: Final joint configuration (7,).
            steps: Number of intermediate waypoint steps.
            method: 'linear' or 'cosine' (smooth S-curve acceleration).
        """
        trajectory = []
        for s in range(steps):
            t = (s + 1) / steps
            if method == "cosine":
                # Cosine blend: 0 -> 1 with zero velocity at endpoints
                alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            else:
                alpha = t
            waypoint = (1.0 - alpha) * start_qpos + alpha * target_qpos
            trajectory.append(waypoint)
        return trajectory
