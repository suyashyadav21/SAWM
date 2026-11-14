"""
6D Pose Inverse Kinematics Solver for Franka Emika Panda in MuJoCo.

Solves both 3D position and 3D orientation using:
- Analytical Jacobians (translational Jp and rotational Jr)
- Damped Least Squares (DLS) formulation
- Nullspace posture projection towards ready_qpos
- Hardware joint limit clamping at every iteration
"""

import numpy as np
import mujoco
from typing import Tuple, Optional


class Franka6DIKSolver:
    """6D Pose Differential Inverse Kinematics solver for Franka Panda."""

    # Default downward grasp orientation: Z_hand points along world -Z
    DEFAULT_GRASP_R = np.array([
        [0.70710678,  0.70710678,  0.0],
        [0.70710678, -0.70710678,  0.0],
        [0.0,         0.0,        -1.0]
    ], dtype=np.float64)

    def __init__(self, model: mujoco.MjModel, config: Optional[dict] = None):
        self.model = model
        cfg = config or {}
        ik_cfg = cfg.get("ik", {})

        self.damping = float(ik_cfg.get("damping", 0.001))
        self.nullspace_gain = float(ik_cfg.get("nullspace_gain", 0.10))
        self.max_iters = int(ik_cfg.get("max_iterations", 100))
        self.tol_pos = float(ik_cfg.get("position_tolerance", 0.001))
        self.tol_rot = float(ik_cfg.get("rotation_tolerance", 0.01))
        self.step_size = float(ik_cfg.get("step_size", 0.50))
        self.rot_weight = float(ik_cfg.get("rot_weight", 0.50))

        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_body_id == -1:
            raise ValueError("Hand body 'hand' not found in MuJoCo model")

        # Cache joint limits for 7 arm joints
        self.arm_joint_limits = [model.jnt_range[j] for j in range(7)]

        # Nominal ready posture for nullspace biasing
        self.ready_qpos = np.array([0.0, -0.4, 0.0, -2.1, 0.0, 1.7, 0.785], dtype=np.float64)

        # Temporary simulation data for solving IK without polluting main simulation
        self._d_sim = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        q_seed: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool, int, float, float]:
        """
        Solve 6D pose IK for end-effector target position and orientation.
        
        Args:
            target_pos: (3,) target world coordinates [x, y, z] in meters.
            target_rot: Optional (3, 3) target rotation matrix. Defaults to DEFAULT_GRASP_R.
            q_seed: Optional (7,) initial joint angle seed in radians.
            
        Returns:
            Tuple of:
                - q_sol: (7,) solved joint angles in radians
                - success: bool indicating convergence within tolerances
                - iterations: int number of iterations executed
                - pos_err: float final Euclidean position error in meters
                - rot_err: float final orientation error in radians
        """
        target_p = np.array(target_pos, dtype=np.float64)
        target_R = target_rot if target_rot is not None else self.DEFAULT_GRASP_R

        # Initialize temporary data with seed
        seed = q_seed if q_seed is not None else self.ready_qpos
        self._d_sim.qpos[:7] = seed
        self._d_sim.qvel[:7] = 0.0
        mujoco.mj_forward(self.model, self._d_sim)

        pos_err_norm = 999.0
        rot_err_norm = 999.0

        for it in range(self.max_iters):
            curr_pos = self._d_sim.xpos[self.hand_body_id]
            curr_R = self._d_sim.xmat[self.hand_body_id].reshape(3, 3)

            # Translation error
            e_p = target_p - curr_pos
            pos_err_norm = float(np.linalg.norm(e_p))

            # Orientation error vector from column cross products
            e_r = 0.5 * (
                np.cross(curr_R[:, 0], target_R[:, 0]) +
                np.cross(curr_R[:, 1], target_R[:, 1]) +
                np.cross(curr_R[:, 2], target_R[:, 2])
            )
            rot_err_norm = float(np.linalg.norm(e_r))

            # Check convergence
            if pos_err_norm < self.tol_pos and rot_err_norm < self.tol_rot:
                return self._d_sim.qpos[:7].copy(), True, it + 1, pos_err_norm, rot_err_norm

            # Compute analytical Jacobians
            mujoco.mj_jacBody(self.model, self._d_sim, self._jacp, self._jacr, self.hand_body_id)
            Jp = self._jacp[:, :7]
            Jr = self._jacr[:, :7]

            # Stack into 6x7 composite Jacobian with rotation weighting
            J = np.vstack([Jp, self.rot_weight * Jr])
            e = np.hstack([e_p, self.rot_weight * e_r])

            # Damped Least Squares inverse
            A = J @ J.T + self.damping * np.eye(6)
            J_dls = J.T @ np.linalg.solve(A, np.eye(6))
            dq_task = J_dls @ e

            # Nullspace posture projection towards ready_qpos
            q_bias = self.ready_qpos - self._d_sim.qpos[:7]
            N = np.eye(7) - J_dls @ J
            dq_null = N @ (self.nullspace_gain * q_bias)

            # Update joint positions
            self._d_sim.qpos[:7] += self.step_size * (dq_task + dq_null)

            # Enforce hardware joint limits
            for j in range(7):
                low, high = self.arm_joint_limits[j]
                self._d_sim.qpos[j] = np.clip(self._d_sim.qpos[j], low, high)

            mujoco.mj_forward(self.model, self._d_sim)

        # Iteration limit reached
        success = (pos_err_norm < self.tol_pos and rot_err_norm < self.tol_rot)
        return self._d_sim.qpos[:7].copy(), success, self.max_iters, pos_err_norm, rot_err_norm
