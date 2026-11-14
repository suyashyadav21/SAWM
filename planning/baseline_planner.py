"""
Baseline Task Planner for Franka Panda Robotics.

Implements conventional, unassisted task-oriented pick-and-place:
- 6D pose inverse kinematics (DLS with nullspace projection)
- 8-stage finite-state machine with explicit failure guards
- Smooth cosine S-curve trajectory interpolation
- Robust grasp verification (contact, lift, and transit attachment tracking)
- Strict baseline independence (SafetyMonitor remains completely passive)
"""

import os
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable

from simulation.environment import SawmEnvironment
from simulation.robot import FrankaPandaRobot
from planning.ik_solver import Franka6DIKSolver
from planning.fsm import PickPlaceFSM, TaskState, TaskResult


class BaselinePlanner:
    """Conventional task-oriented baseline planner for pick-and-place."""

    def __init__(self, env: SawmEnvironment, config_path: str = "configs/planner.yaml"):
        self.env = env
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize sub-modules
        self.ik_solver = Franka6DIKSolver(env.model, self.config)
        self.fsm = PickPlaceFSM(self.config)

        # Gripper control values from config
        self.open_ctrl = float(self.config.get("gripper", {}).get("open_ctrl", 255.0))
        self.close_ctrl = float(self.config.get("gripper", {}).get("close_ctrl", 0.0))

        # FSM parameters
        fsm_cfg = self.config.get("fsm", {})
        self.table_z = float(fsm_cfg.get("table_surface_z", 0.400))
        self.pregrasp_z = float(fsm_cfg.get("pregrasp_z", 0.620))
        self.grasp_z = float(fsm_cfg.get("grasp_descent_z", 0.525))
        self.lift_z = float(fsm_cfg.get("lift_z", 0.650))
        self.place_z = float(fsm_cfg.get("place_z", 0.535))

        self.transit_steps = int(fsm_cfg.get("transit_steps", 70))
        self.vertical_steps = int(fsm_cfg.get("vertical_steps", 50))
        self.grasp_dwell = int(fsm_cfg.get("grasp_dwell_steps", 40))
        self.release_dwell = int(fsm_cfg.get("release_dwell_steps", 30))

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def plan_keyframes(self, obj_pos: np.ndarray, goal_pos: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool, str]:
        """
        Compute joint configurations for all keyframe stages using 6D IK.
        
        Returns:
            Tuple of:
                - keyframes: dict of stage_name -> qpos (7,)
                - success: bool True if all IK solves converged
                - error_msg: diagnostic string if any IK failed
        """
        keyframes = {"home": self.env.home_qpos.copy()}

        # Define 6D Cartesian targets
        targets = {
            "pregrasp": np.array([obj_pos[0], obj_pos[1], self.pregrasp_z]),
            "grasp": np.array([obj_pos[0], obj_pos[1], self.grasp_z]),
            "lift": np.array([obj_pos[0], obj_pos[1], self.lift_z]),
            "transport": np.array([goal_pos[0], goal_pos[1], self.lift_z]),
            "place": np.array([goal_pos[0], goal_pos[1], self.place_z]),
            "retract": np.array([goal_pos[0], goal_pos[1], self.lift_z]),
        }

        q_seed = self.env.ready_qpos
        for stage_name, target_pos in targets.items():
            q_sol, success, iters, pos_err, rot_err = self.ik_solver.solve(target_pos, q_seed=q_seed)
            if not success:
                return {}, False, f"IK_CONVERGENCE_FAILURE: Failed to converge for '{stage_name}' (pos_err={pos_err*1000:.1f}mm)"
            keyframes[stage_name] = q_sol
            q_seed = q_sol  # Chain seed for smooth transitions

        return keyframes, True, ""

    def execute(self, step_callback: Optional[Callable] = None) -> TaskResult:
        """
        Execute the pick-and-place task to completion or failure.
        
        Args:
            step_callback: Optional func(step_idx, obs, fsm_state) called each control step.
            
        Returns:
            TaskResult containing outcome, timing, path length, and placement error.
        """
        self.fsm.reset()
        obs = self.env.reset()

        obj_pos = obs["target_object"]["position"]
        goal_pos = obs["goal"]["position"]

        # 1. Compute keyframes
        keyframes, ik_success, ik_error = self.plan_keyframes(obj_pos, goal_pos)
        if not ik_success:
            self.fsm.fail(ik_error)
            return TaskResult(
                success=False,
                final_state=self.fsm.state.value,
                failure_reason=ik_error,
                state_history=self.fsm.state_history,
            )

        # 2. Plan sequence of trajectory segments
        segments = [
            (TaskState.HOME, keyframes["home"], keyframes["home"], 10, self.open_ctrl),
            (TaskState.MOVE_TO_PREGRASP, keyframes["home"], keyframes["pregrasp"], self.transit_steps, self.open_ctrl),
            (TaskState.DESCEND_TO_OBJECT, keyframes["pregrasp"], keyframes["grasp"], self.vertical_steps, self.open_ctrl),
            (TaskState.GRASP, keyframes["grasp"], keyframes["grasp"], self.grasp_dwell, self.close_ctrl),
            (TaskState.LIFT, keyframes["grasp"], keyframes["lift"], self.vertical_steps, self.close_ctrl),
            (TaskState.MOVE_TO_TARGET, keyframes["lift"], keyframes["transport"], self.transit_steps + 20, self.close_ctrl),
            (TaskState.RELEASE, keyframes["transport"], keyframes["place"], self.vertical_steps, self.close_ctrl),
            (TaskState.RELEASE, keyframes["place"], keyframes["place"], self.release_dwell, self.open_ctrl),
            (TaskState.RETURN_HOME, keyframes["place"], keyframes["retract"], self.vertical_steps, self.open_ctrl),
            (TaskState.RETURN_HOME, keyframes["retract"], keyframes["home"], self.transit_steps, self.open_ctrl),
        ]

        # Execution tracking metrics
        prev_ee_pos = obs["robot"]["ee_position"].copy()
        cumulative_path_len = 0.0
        joint_limit_violations = 0
        step_idx = 0
        start_time = obs["time"]

        # Passive safety monitoring telemetry
        safety_states = {"SAFE": 0, "WARNING": 0, "DANGER": 0}
        min_clearance = float("inf")
        collisions_observed = 0
        near_misses_observed = 0

        for expected_state, q_start, q_end, num_steps, gripper_cmd in segments:
            if self.fsm.is_terminal:
                break

            self.fsm.transition_to(expected_state)
            traj = FrankaPandaRobot.interpolate_trajectory(q_start, q_end, steps=num_steps, method="cosine")

            for wp in traj:
                step_idx += 1
                self.fsm.state_step_count += 1

                # Step simulation (Note: baseline commands arm strictly without safety intervention)
                obs = self.env.step(arm_targets=wp, gripper_target=gripper_cmd)

                # Track passive safety telemetry (zero intervention on control)
                if "safety" in obs:
                    s_state = obs["safety"].state
                    safety_states[s_state] = safety_states.get(s_state, 0) + 1
                    if hasattr(obs["safety"], "collision_count"):
                        collisions_observed = max(collisions_observed, obs["safety"].collision_count)
                    if hasattr(obs["safety"], "near_miss_count"):
                        near_misses_observed = max(near_misses_observed, obs["safety"].near_miss_count)
                if "distances" in obs:
                    d_min = obs["distances"].geometric_min_distance
                    if d_min is not None and not np.isnan(d_min) and d_min < 900.0:
                        min_clearance = min(min_clearance, float(d_min))

                # Track trajectory length
                curr_ee = obs["robot"]["ee_position"]
                cumulative_path_len += float(np.linalg.norm(curr_ee - prev_ee_pos))
                prev_ee_pos = curr_ee.copy()

                # Track joint limits
                if any(v["violated"] for v in obs["joint_limits"].values()):
                    joint_limit_violations += 1

                # FSM state checks & failure guards
                curr_obj = obs["target_object"]["position"]
                contacts = obs["contacts"]

                # Grasp / slippage verification
                if self.fsm.state in [TaskState.LIFT, TaskState.MOVE_TO_TARGET]:
                    valid = self.fsm.update_grasp_tracking(curr_ee, curr_obj, contacts)
                    if not valid:
                        break

                # Timeout check
                if self.fsm.check_timeout():
                    break

                if step_callback:
                    step_callback(step_idx, obs, self.fsm.state.value)

            # End of LIFT segment: strict grasp confirmation
            if expected_state == TaskState.LIFT and not self.fsm.is_terminal:
                lift_height = obs["target_object"]["position"][2] - self.table_z
                has_contact = contacts.get("robot_object_contact", False) if isinstance(contacts, dict) else getattr(contacts, "robot_object_contact", False)
                if not (lift_height >= self.fsm.min_lift_height and has_contact):
                    self.fsm.fail(f"GRASP_FAILURE: Object not lifted (height={lift_height*1000:.1f}mm, contact={has_contact})")
                    break

        # 3. Final Placement & Completion Verification
        final_obs = self.env.get_observation()
        final_obj = final_obs["target_object"]["position"]
        final_goal = final_obs["goal"]["position"]

        is_placed, place_err = self.fsm.verify_placement(final_obj, final_goal)

        if not self.fsm.is_terminal:
            if is_placed:
                self.fsm.transition_to(TaskState.COMPLETED)
            else:
                self.fsm.fail(f"PLACEMENT_FAILURE: Object placed {place_err*1000:.1f}mm from goal (radius={self.fsm.goal_radius*1000:.1f}mm)")

        total_exec_time = final_obs["time"] - start_time

        safety_summary = {
            "safety_states": safety_states,
            "min_clearance": min_clearance if min_clearance != float("inf") else None,
            "collisions": collisions_observed,
            "near_misses": near_misses_observed,
        }

        return TaskResult(
            success=self.fsm.is_success,
            final_state=self.fsm.state.value,
            failure_reason=self.fsm.failure_reason,
            execution_time=total_exec_time,
            trajectory_length=cumulative_path_len,
            grasp_success=self.fsm.grasp_verified,
            placement_error=place_err,
            joint_limit_violations=joint_limit_violations,
            state_history=self.fsm.state_history,
            stages_completed=self.fsm.stages_completed,
            safety_summary=safety_summary,
        )
