"""
Scenario Manager for SAWM Milestone 2.

Orchestrates the 5 required safety scenarios:
1. Normal Operation (no hazards active, status SAFE throughout)
2. Obstacle Near Motion Path (proximity warning, status SAFE -> WARNING -> SAFE)
3. Restricted Zone Approach (contextual ZONE_BREACH -> DANGER, collision=False)
4. Moving Proxy Encroachment (danger distance breach d < 0.08m, near-miss count increment, recovery)
5. Physical Collision Event (contact detected, DANGER latched until reset)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

from simulation.environment import SawmEnvironment
from simulation.robot import FrankaPandaRobot


class ScenarioManager:
    """Configures and runs deterministic safety evaluation scenarios."""

    def __init__(self, env: SawmEnvironment):
        self.env = env
        self.scenarios_cfg = env.safety_config.get("scenarios", {})

    def setup_scenario(self, scenario_id: int) -> dict:
        """
        Configure the environment for a specific scenario ID (1 to 5).
        
        Args:
            scenario_id: Integer from 1 to 5.
            
        Returns:
            Scenario metadata dictionary.
        """
        # Always start from fresh clean reset
        self.env.reset()
        self.env.obstacle_manager.park_all()
        self.env.restricted_zones_enabled = False

        if scenario_id == 1:
            # Scenario 1: Normal Operation
            meta = {
                "id": 1,
                "name": "Normal Operation",
                "description": "Standard nominal trajectory with zero hazards in workspace. Expected: SAFE throughout.",
            }
            return meta

        elif scenario_id == 2:
            # Scenario 2: Obstacle Near Motion Path
            s_cfg = self.scenarios_cfg.get("scenario2_obstacle_path", {})
            pos = np.array(s_cfg.get("static_obstacle_pos", [0.65, 0.18, 0.52]), dtype=np.float64)
            self.env.obstacle_manager.set_static_obstacle_position(pos, active=True)
            meta = {
                "id": 2,
                "name": "Obstacle Near Motion Path",
                "description": f"Static pillar placed at {pos.tolist()} (0.12m clearance from transit path). Expected: SAFE -> WARNING -> SAFE.",
                "obstacle_pos": pos,
            }
            return meta

        elif scenario_id == 3:
            # Scenario 3: Restricted Zone Approach
            self.env.restricted_zones_enabled = True
            meta = {
                "id": 3,
                "name": "Restricted Zone Approach",
                "description": "Robot trajectory enters operator keep-out volume. Expected: Contextual ZONE_BREACH -> DANGER.",
            }
            return meta

        elif scenario_id == 4:
            # Scenario 4: Moving Proxy Encroachment
            s_cfg = self.scenarios_cfg.get("scenario4_danger_proximity", {})
            start_pos = np.array(s_cfg.get("human_proxy_start", [0.85, 0.25, 0.60]), dtype=np.float64)
            target_pos = np.array(s_cfg.get("human_proxy_target", [0.52, 0.12, 0.58]), dtype=np.float64)
            # Setup two-way trajectory (approach in 60 steps, hold in 30 steps, retract in 60 steps)
            self.env.obstacle_manager.set_proxy_trajectory(start_pos, target_pos, steps=60)
            meta = {
                "id": 4,
                "name": "Moving Proxy Encroachment",
                "description": f"Kinematic arm proxy moves from {start_pos.tolist()} to {target_pos.tolist()} (d < 0.08m). Expected: DANGER -> Recovery.",
                "proxy_start": start_pos,
                "proxy_target": target_pos,
            }
            return meta

        elif scenario_id == 5:
            # Scenario 5: Physical Collision Event
            s_cfg = self.scenarios_cfg.get("scenario5_collision_event", {})
            pos = np.array(s_cfg.get("static_obstacle_pos", [0.45, 0.00, 0.51]), dtype=np.float64)
            self.env.obstacle_manager.set_static_obstacle_position(pos, active=True)
            meta = {
                "id": 5,
                "name": "Physical Collision Event",
                "description": f"Static obstacle placed directly in descent corridor at {pos.tolist()}. Expected: Collision -> Latched DANGER.",
                "obstacle_pos": pos,
            }
            return meta

        else:
            raise ValueError(f"Unknown scenario ID: {scenario_id}. Choose 1 through 5.")

    def run_scenario(self, scenario_id: int, step_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Execute a full scenario episode and record comprehensive telemetry.
        
        Args:
            scenario_id: Integer 1 to 5.
            step_callback: Optional callback func(step_idx, obs) invoked each control step.
            
        Returns:
            Dictionary containing episode summary and time-series telemetry.
        """
        meta = self.setup_scenario(scenario_id)
        env = self.env
        robot = env.robot

        # Multi-stage trajectory definition based on scenario
        if scenario_id == 3:
            # Zone intrusion trajectory: arm swings towards restricted operator zone
            s3_cfg = self.scenarios_cfg.get("scenario3_restricted_zone", {})
            zone_target_q = np.array(s3_cfg.get("robot_target_pose", [-0.40, -0.15, 0.0, -2.1, 0.0, 1.8, 0.785]), dtype=np.float64)
            waypoints = [
                ("Home -> Ready Pose", env.home_qpos, env.ready_qpos, 40),
                ("Ready -> Restricted Zone Intrusion", env.ready_qpos, zone_target_q, 60),
                ("Hold in Zone", zone_target_q, zone_target_q, 30),
                ("Retract to Ready Pose", zone_target_q, env.ready_qpos, 60),
            ]
        elif scenario_id == 4:
            # Moving proxy: robot moves to Ready pose while proxy advances
            waypoints = [
                ("Home -> Ready Pose (Proxy Inbound)", env.home_qpos, env.ready_qpos, 60),
                ("Hold at Ready (Proxy Encroaches)", env.ready_qpos, env.ready_qpos, 60),
                ("Retract Proxy to Safe Distance", env.ready_qpos, env.ready_qpos, 60),
            ]
        elif scenario_id == 2:
            # Scenario 2: Obstacle near path. Demonstrates SAFE -> WARNING -> SAFE recovery
            waypoints = [
                ("Home -> Ready Pose", env.home_qpos, env.ready_qpos, 50),
                ("Ready -> Approach Corridor", env.ready_qpos, env.approach_qpos, 50),
                ("Hold in Proximity", env.approach_qpos, env.approach_qpos, 30),
                ("Retract to Home Pose (Clearance Recovery)", env.approach_qpos, env.home_qpos, 50),
            ]
        else:
            # Nominal trajectory for scenarios 1 and 5
            waypoints = [
                ("Home -> Ready Pose", env.home_qpos, env.ready_qpos, 50),
                ("Ready -> Approach Corridor", env.ready_qpos, env.approach_qpos, 50),
                ("Hold / Grasp Action", env.approach_qpos, env.approach_qpos, 30),
                ("Retract to Ready Pose", env.approach_qpos, env.ready_qpos, 50),
            ]

        # Telemetry history
        history = {
            "time": [],
            "state": [],
            "geometric_min_distance": [],
            "link_center_min_distance": [],
            "physical_risk": [],
            "contextual_harm": [],
            "collision_detected": [],
            "is_latched": [],
        }

        step_idx = 0
        min_geom_dist_seen = 999.0
        max_phys_risk_seen = 0.0
        max_ctx_harm_seen = 0.0
        states_seen = set()

        for stage_name, start_q, target_q, num_steps in waypoints:
            # For scenario 4: trigger proxy return in the 3rd stage
            if scenario_id == 4 and "Retract" in stage_name:
                proxy_cur = env.obstacle_manager.get_proxy_position()
                proxy_park = env.obstacle_manager.proxy_parked_pos
                env.obstacle_manager.set_proxy_trajectory(proxy_cur, proxy_park, steps=num_steps)

            traj = FrankaPandaRobot.interpolate_trajectory(start_q, target_q, steps=num_steps, method="cosine")

            for wp in traj:
                step_idx += 1
                obs = env.step(arm_targets=wp)

                t = obs["time"]
                safety = obs["safety"]
                dists = obs["distances"]
                contacts = obs["contacts"]

                g_dist = dists.geometric_min_distance
                c_dist = dists.link_center_min_distance
                state = safety.state
                states_seen.add(state)

                if g_dist < min_geom_dist_seen:
                    min_geom_dist_seen = g_dist
                if safety.physical_risk_score > max_phys_risk_seen:
                    max_phys_risk_seen = safety.physical_risk_score
                if safety.contextual_harm_score > max_ctx_harm_seen:
                    max_ctx_harm_seen = safety.contextual_harm_score

                is_col = contacts["robot_table_collision"] or contacts["robot_object_contact"] or \
                         obs["safety"].collision_count > 0

                history["time"].append(t)
                history["state"].append(state)
                history["geometric_min_distance"].append(g_dist)
                history["link_center_min_distance"].append(c_dist)
                history["physical_risk"].append(safety.physical_risk_score)
                history["contextual_harm"].append(safety.contextual_harm_score)
                history["collision_detected"].append(safety.collision_count > 0)
                history["is_latched"].append(safety.is_latched)

                if step_callback:
                    step_callback(step_idx, obs, meta)

        final_obs = env.get_observation()
        summary = {
            "meta": meta,
            "total_steps": step_idx,
            "sim_time": final_obs["time"],
            "states_seen": sorted(list(states_seen)),
            "final_state": final_obs["safety"].state,
            "final_is_latched": final_obs["safety"].is_latched,
            "min_geometric_distance": min_geom_dist_seen,
            "max_physical_risk": max_phys_risk_seen,
            "max_contextual_harm": max_ctx_harm_seen,
            "collision_count": final_obs["safety"].collision_count,
            "near_miss_count": final_obs["safety"].near_miss_count,
            "history": history,
        }
        return summary
