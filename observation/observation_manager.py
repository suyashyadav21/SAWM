"""
Unified Observation Manager for SAWM.

Tracks:
- All 11 robot bodies (link0-link7, hand, left_finger, right_finger)
- Exact surface-to-surface geometric distance (mj_geomDistance)
- Approximate link-center clearance
- Contextual restricted zone intrusion using true Cartesian world coordinates
- Comprehensive contact and collision diagnostics
"""

import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional, Any

from observation.state import (
    RobotState,
    HazardState,
    RestrictedZoneState,
    DistanceMetrics,
    ContactMetrics,
)
from simulation.robot import FrankaPandaRobot
from simulation.obstacles import ObstacleManager


class ObservationManager:
    """Extracts and computes structured physical and contextual observations."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, safety_config: dict):
        self.model = model
        self.data = data
        self.config = safety_config

        # ---------------------------------------------------------------------
        # 1. Track All 11 Robot Bodies & Associated Geoms
        # ---------------------------------------------------------------------
        self.robot_body_names = [f"link{i}" for i in range(8)] + ["hand", "left_finger", "right_finger"]
        self.robot_body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.robot_body_names}

        # Cache robot collision geoms (contype == 1)
        self.robot_collision_geoms: List[Tuple[str, int, str]] = []  # (geom_name, geom_id, parent_body_name)
        for g_id in range(model.ngeom):
            b_id = model.geom_bodyid[g_id]
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)
            if b_name in self.robot_body_names and model.geom_contype[g_id] == 1:
                g_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or f"geom_{g_id}"
                self.robot_collision_geoms.append((g_name, g_id, b_name))

        # ---------------------------------------------------------------------
        # 2. Restricted Zone Definitions
        # ---------------------------------------------------------------------
        self.zones_cfg = self.config.get("hazards", {}).get("restricted_zones", [])
        self.restricted_zones = []
        for z in self.zones_cfg:
            self.restricted_zones.append({
                "name": z["name"],
                "bounds_min": np.array(z["bounds_min"], dtype=np.float64),
                "bounds_max": np.array(z["bounds_max"], dtype=np.float64),
                "warning_buffer": float(z.get("warning_buffer", 0.05)),
            })

        # Buffer for geom distance query
        self._fromto_buf = np.zeros(6, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Observation Collection
    # -------------------------------------------------------------------------
    def collect_robot_state(self, robot: FrankaPandaRobot) -> RobotState:
        """Collect kinematic state across all 11 robot bodies."""
        qpos = robot.get_joint_positions()
        qvel = robot.get_joint_velocities()
        ee_pos, ee_quat = robot.get_end_effector_pose()
        gripper_pos = robot.get_gripper_position()

        body_positions = {}
        for name, b_id in self.robot_body_ids.items():
            if b_id != -1:
                body_positions[name] = np.array(self.data.xpos[b_id], dtype=np.float64)

        return RobotState(
            joint_positions=qpos,
            joint_velocities=qvel,
            ee_position=ee_pos,
            ee_quaternion=ee_quat,
            gripper_position=gripper_pos,
            body_positions=body_positions,
        )

    def compute_distances(self, obstacle_mgr: ObstacleManager) -> DistanceMetrics:
        """
        Compute minimum separation distances.
        Explicitly distinguishes exact geometric distance from link-center clearance.
        """
        active_hazards = obstacle_mgr.get_active_hazard_geoms()
        if not active_hazards:
            return DistanceMetrics()

        # A. Exact Geometric Surface Distance via MuJoCo GJK/EPA solver
        min_geom_dist = 999.0
        nearest_r_geom = "none"
        nearest_h_geom = "none"
        nearest_pts = None

        for r_name, r_gid, r_bname in self.robot_collision_geoms:
            for h_name, h_gid in active_hazards:
                dist = mujoco.mj_geomDistance(self.model, self.data, r_gid, h_gid, 2.0, self._fromto_buf)
                if dist < min_geom_dist:
                    min_geom_dist = float(dist)
                    nearest_r_geom = f"{r_bname}:{r_name}"
                    nearest_h_geom = h_name
                    nearest_pts = (self._fromto_buf[:3].copy(), self._fromto_buf[3:].copy())

        # B. Approximate Link-Center Clearance
        min_center_dist = 999.0
        nearest_r_body = "none"
        nearest_h_body = "none"

        hazard_positions = []
        if obstacle_mgr.is_static_active():
            hazard_positions.append(("static_obstacle", obstacle_mgr.get_static_obstacle_position(), 0.04))
        if obstacle_mgr.is_proxy_active():
            hazard_positions.append(("human_proxy", obstacle_mgr.get_proxy_position(), 0.035))

        for b_name, b_id in self.robot_body_ids.items():
            if b_id == -1:
                continue
            r_pos = self.data.xpos[b_id]
            for h_name, h_pos, h_radius in hazard_positions:
                center_dist = float(np.linalg.norm(r_pos - h_pos)) - h_radius
                if center_dist < min_center_dist:
                    min_center_dist = center_dist
                    nearest_r_body = b_name
                    nearest_h_body = h_name

        return DistanceMetrics(
            geometric_min_distance=min_geom_dist,
            nearest_robot_geom=nearest_r_geom,
            nearest_hazard_geom=nearest_h_geom,
            nearest_points=nearest_pts,
            link_center_min_distance=min_center_dist,
            nearest_robot_body=nearest_r_body,
            nearest_hazard_body=nearest_h_body,
        )

    def evaluate_restricted_zones(self, zone_enabled: bool = True) -> List[RestrictedZoneState]:
        """
        Evaluate keep-out zone intrusions using true Cartesian world coordinates
        of robot link bodies and collision geoms. Rotational joint angles (qpos) are strictly excluded.
        """
        zone_states = []
        if not zone_enabled:
            for z in self.restricted_zones:
                zone_states.append(RestrictedZoneState(
                    name=z["name"],
                    bounds_min=z["bounds_min"],
                    bounds_max=z["bounds_max"],
                    warning_buffer=z["warning_buffer"],
                    breached=False,
                    warning_active=False,
                    min_distance_to_zone=999.0,
                ))
            return zone_states

        for z in self.restricted_zones:
            b_min = z["bounds_min"]
            b_max = z["bounds_max"]
            buf = z["warning_buffer"]
            warn_min = b_min - buf
            warn_max = b_max + buf

            breached = False
            warning_active = False
            min_dist = 999.0
            breaching_bodies = []

            # Check all 11 robot bodies (using actual Cartesian world positions data.xpos)
            for b_name, b_id in self.robot_body_ids.items():
                if b_id == -1:
                    continue
                pos = self.data.xpos[b_id]

                # Point-in-box check for strict breach
                in_box = np.all(pos >= b_min) and np.all(pos <= b_max)
                if in_box:
                    breached = True
                    breaching_bodies.append(b_name)

                # Point-in-box check for warning buffer
                in_warn = np.all(pos >= warn_min) and np.all(pos <= warn_max)
                if in_warn:
                    warning_active = True

                # Compute distance from point to AABB
                dx = max(b_min[0] - pos[0], 0.0, pos[0] - b_max[0])
                dy = max(b_min[1] - pos[1], 0.0, pos[1] - b_max[1])
                dz = max(b_min[2] - pos[2], 0.0, pos[2] - b_max[2])
                d_box = float(np.sqrt(dx*dx + dy*dy + dz*dz))
                if in_box:
                    d_box = 0.0
                min_dist = min(min_dist, d_box)

            zone_states.append(RestrictedZoneState(
                name=z["name"],
                bounds_min=b_min,
                bounds_max=b_max,
                warning_buffer=buf,
                breached=breached,
                warning_active=warning_active,
                min_distance_to_zone=min_dist,
                breaching_bodies=breaching_bodies,
            ))

        return zone_states

    def collect_contacts(self) -> ContactMetrics:
        """Inspect all active contacts and categorize by hazard interaction."""
        raw_contacts = []
        robot_names = set(self.robot_body_names)

        robot_obs_col = False
        robot_proxy_col = False
        robot_table_col = False
        robot_obj_contact = False
        max_penetration = 0.0

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            b1_id = self.model.geom_bodyid[g1]
            b2_id = self.model.geom_bodyid[g2]
            b1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1_id) or f"body_{b1_id}"
            b2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2_id) or f"body_{b2_id}"

            is_b1_robot = b1 in robot_names
            is_b2_robot = b2 in robot_names

            if not (is_b1_robot or is_b2_robot):
                continue

            dist = float(con.dist)
            if dist < 0:
                max_penetration = max(max_penetration, abs(dist))

            other_body = b2 if is_b1_robot else b1
            robot_body = b1 if is_b1_robot else b2

            if "static_obstacle" in other_body:
                robot_obs_col = True
            elif "human_proxy" in other_body:
                robot_proxy_col = True
            elif "table" in other_body:
                robot_table_col = True
            elif "target_object" in other_body:
                robot_obj_contact = True

            raw_contacts.append({
                "robot_body": robot_body,
                "other_body": other_body,
                "dist": dist,
                "pos": np.array(con.pos, dtype=np.float64),
            })

        return ContactMetrics(
            total_contacts=len(raw_contacts),
            robot_obstacle_collision=robot_obs_col,
            robot_proxy_collision=robot_proxy_col,
            robot_table_collision=robot_table_col,
            robot_object_contact=robot_obj_contact,
            max_penetration=max_penetration,
            contact_pairs=raw_contacts,
        )
