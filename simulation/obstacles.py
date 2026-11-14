"""
Obstacle and Dynamic Hazard Manager for MuJoCo.

Manages:
- Static obstacle placement and activation
- Dynamic moving hazard proxy (simplified kinematic arm/hand representation)
- Smooth kinematic trajectory updates via MuJoCo mocap bodies
- Parking and activating hazards according to scenario specifications
"""

import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple, Any


class ObstacleManager:
    """Manages static and dynamic hazards using MuJoCo mocap bodies."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        self.model = model
        self.data = data
        self.config = config.get("hazards", {})

        # Identify mocap bodies
        self.static_obs_cfg = self.config.get("static_obstacle", {})
        self.proxy_cfg = self.config.get("human_proxy", {})

        # Static obstacle
        self.static_body_name = self.static_obs_cfg.get("body_name", "static_obstacle")
        self.static_geom_name = self.static_obs_cfg.get("geom_name", "static_obstacle_geom")
        self.static_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.static_body_name)
        self.static_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.static_geom_name)
        self.static_mocap_id = model.body_mocapid[self.static_body_id] if self.static_body_id != -1 else -1

        # Moving proxy (simplified kinematic arm/hand representation)
        self.proxy_body_name = self.proxy_cfg.get("body_name", "human_proxy")
        self.proxy_geom_name = self.proxy_cfg.get("geom_name", "human_proxy_geom")
        self.proxy_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.proxy_body_name)
        self.proxy_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.proxy_geom_name)
        self.proxy_mocap_id = model.body_mocapid[self.proxy_body_id] if self.proxy_body_id != -1 else -1

        # Default parked positions (offscreen/inactive)
        self.static_parked_pos = np.array(self.static_obs_cfg.get("parked_pos", [2.0, 0.0, -1.0]), dtype=np.float64)
        self.proxy_parked_pos = np.array(self.proxy_cfg.get("parked_pos", [2.0, 0.0, -1.0]), dtype=np.float64)

        # Dynamic motion state
        self._proxy_start_pos = self.proxy_parked_pos.copy()
        self._proxy_target_pos = self.proxy_parked_pos.copy()
        self._proxy_step_idx = 0
        self._proxy_total_steps = 1
        self._proxy_active = False

        # Static active state
        self._static_active = False
        self._static_active_pos = np.array(self.static_obs_cfg.get("default_active_pos", [0.48, -0.05, 0.55]), dtype=np.float64)

        # Park all by default
        self.park_all()

    # -------------------------------------------------------------------------
    # Static Obstacle Control
    # -------------------------------------------------------------------------
    def set_static_obstacle_position(self, pos: np.ndarray, active: bool = True) -> None:
        """Place static obstacle at specified world coordinates."""
        if self.static_mocap_id != -1:
            self.data.mocap_pos[self.static_mocap_id] = pos
            self._static_active = active
            self._static_active_pos = np.array(pos, dtype=np.float64)

    def park_static_obstacle(self) -> None:
        """Move static obstacle to inert parked location."""
        if self.static_mocap_id != -1:
            self.data.mocap_pos[self.static_mocap_id] = self.static_parked_pos
            self._static_active = False

    # -------------------------------------------------------------------------
    # Moving Proxy Control (Simplified Kinematic Arm/Hand)
    # -------------------------------------------------------------------------
    def set_proxy_trajectory(self, start_pos: np.ndarray, target_pos: np.ndarray, steps: int) -> None:
        """Initialize a smooth kinematic trajectory for the moving proxy."""
        self._proxy_start_pos = np.array(start_pos, dtype=np.float64)
        self._proxy_target_pos = np.array(target_pos, dtype=np.float64)
        self._proxy_total_steps = max(1, steps)
        self._proxy_step_idx = 0
        self._proxy_active = True
        if self.proxy_mocap_id != -1:
            self.data.mocap_pos[self.proxy_mocap_id] = self._proxy_start_pos

    def step_proxy_trajectory(self) -> np.ndarray:
        """Advance moving proxy one step along its trajectory using cosine interpolation."""
        if not self._proxy_active or self.proxy_mocap_id == -1:
            return self.get_proxy_position()

        self._proxy_step_idx = min(self._proxy_step_idx + 1, self._proxy_total_steps)
        t = self._proxy_step_idx / self._proxy_total_steps
        # Cosine blending for smooth acceleration and deceleration
        alpha = 0.5 * (1.0 - np.cos(np.pi * t))
        current_pos = (1.0 - alpha) * self._proxy_start_pos + alpha * self._proxy_target_pos

        self.data.mocap_pos[self.proxy_mocap_id] = current_pos
        return current_pos

    def park_proxy(self) -> None:
        """Move kinematic proxy to inert parked location."""
        self._proxy_active = False
        if self.proxy_mocap_id != -1:
            self.data.mocap_pos[self.proxy_mocap_id] = self.proxy_parked_pos

    def park_all(self) -> None:
        """Park all hazards out of the workspace."""
        self.park_static_obstacle()
        self.park_proxy()

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------
    def get_static_obstacle_position(self) -> np.ndarray:
        if self.static_body_id != -1:
            return np.array(self.data.xpos[self.static_body_id], dtype=np.float64)
        return self.static_parked_pos.copy()

    def get_proxy_position(self) -> np.ndarray:
        if self.proxy_body_id != -1:
            return np.array(self.data.xpos[self.proxy_body_id], dtype=np.float64)
        return self.proxy_parked_pos.copy()

    def is_static_active(self) -> bool:
        return self._static_active

    def is_proxy_active(self) -> bool:
        return self._proxy_active

    def get_active_hazard_geoms(self) -> List[Tuple[str, int]]:
        """Return list of (geom_name, geom_id) for active hazards in the arena."""
        active = []
        if self._static_active and self.static_geom_id != -1:
            active.append((self.static_geom_name, self.static_geom_id))
        if self._proxy_active and self.proxy_geom_id != -1:
            active.append((self.proxy_geom_name, self.proxy_geom_id))
        return active
