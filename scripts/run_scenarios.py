"""
SAWM - Milestone 2 Demonstration Runner: Hazards & Safety Scenarios.

Executes and visualizes the 5 required safety scenarios:
  1: Normal Operation (All hazards parked, SAFE throughout)
  2: Obstacle Near Motion Path (Proximity warning, SAFE -> WARNING -> SAFE)
  3: Restricted Zone Approach (Contextual keep-out breach, ZONE_BREACH -> DANGER)
  4: Moving Proxy Encroachment (Danger distance breach d < 0.08m, near-miss increment, recovery)
  5: Physical Collision Event (Contact detected, DANGER latched until reset)

Usage:
  python scripts/run_scenarios.py --scenario 2
  python scripts/run_scenarios.py --all
  python scripts/run_scenarios.py --scenario 4 --view
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.scenarios import ScenarioManager
from simulation.utils import save_png


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 2 - Hazard & Safety Scenarios Runner")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4, 5], default=2, help="Scenario ID (1 to 5)")
    parser.add_argument("--all", action="store_true", help="Run all 5 scenarios sequentially")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D viewer")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for screenshots")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier in viewer")
    return parser.parse_args()


def format_telemetry_line(step: int, obs: dict) -> str:
    t = obs["time"]
    safety = obs["safety"]
    dists = obs["distances"]
    ee_pos = obs["robot"]["ee_position"]

    g_dist_str = f"{dists.geometric_min_distance:.3f}m" if dists.geometric_min_distance < 900.0 else "N/A"
    c_dist_str = f"{dists.link_center_min_distance:.3f}m" if dists.link_center_min_distance < 900.0 else "N/A"

    state_badge = f"[{safety.state}]"
    if safety.is_latched:
        state_badge += "(LATCHED)"

    alarms_str = f" | {safety.active_alarms[0]}" if safety.active_alarms else ""

    return (
        f"[{t:5.2f}s] EE:[{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] | "
        f"d_geom:{g_dist_str:>7} | d_center:{c_dist_str:>7} | "
        f"Risk:{safety.physical_risk_score:.2f} Harm:{safety.contextual_harm_score:.2f} | "
        f"{state_badge:<17}{alarms_str}"
    )


def run_single_scenario(scenario_id: int, env: SawmEnvironment, args, viewer=None):
    mgr = ScenarioManager(env)
    meta = mgr.setup_scenario(scenario_id)

    print("\n" + "=" * 80)
    print(f"SCENARIO {scenario_id}: {meta['name'].upper()}")
    print(f"Description: {meta['description']}")
    print("=" * 80)

    keyframe_step = 0
    keyframe_frame = None

    def on_step(step_idx, obs, meta_dict):
        nonlocal keyframe_step, keyframe_frame
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep((env.dt * env.frame_skip) / max(0.1, args.speed))

        if step_idx % 15 == 0 or obs["safety"].state in ["WARNING", "DANGER"]:
            print(format_telemetry_line(step_idx, obs))

        # Capture keyframe during peak hazard event
        if obs["safety"].state in ["WARNING", "DANGER"] and keyframe_frame is None:
            keyframe_frame = env.render(camera="workspace_cam", width=1280, height=720)
            keyframe_step = step_idx

    res = mgr.run_scenario(scenario_id, step_callback=on_step)

    # If no keyframe was captured during peak, capture final frame
    if keyframe_frame is None:
        keyframe_frame = env.render(camera="workspace_cam", width=1280, height=720)

    # Save screenshot
    os.makedirs(args.output_dir, exist_ok=True)
    shot_path = os.path.join(args.output_dir, f"milestone2_scenario{scenario_id}.png")
    save_png(shot_path, keyframe_frame)
    file_size_kb = os.path.getsize(shot_path) / 1024.0

    print("-" * 80)
    print(f"Scenario {scenario_id} Summary:")
    print(f"  States Encounted:        {res['states_seen']}")
    print(f"  Final State:             {res['final_state']} (Latched: {res['final_is_latched']})")
    print(f"  Min Geometric Distance:  {res['min_geometric_distance']:.3f} m")
    print(f"  Max Physical Risk Score: {res['max_physical_risk']:.2f}")
    print(f"  Max Contextual Harm:     {res['max_contextual_harm']:.2f}")
    print(f"  Near-Miss Count:         {res['near_miss_count']}")
    print(f"  Collision Count:         {res['collision_count']}")
    print(f"  Screenshot Saved:        {shot_path} ({file_size_kb:.1f} KB)")
    print("=" * 80)
    return res


def main():
    args = parse_args()

    print("=" * 80)
    print("SAWM - Milestone 2 Hazard & Safety Scenarios Demonstration")
    print("=" * 80)

    env = SawmEnvironment()
    viewer = None
    if args.view:
        import mujoco.viewer
        print("[INFO] Launching interactive MuJoCo passive viewer...")
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        if args.all:
            results = []
            for s_id in [1, 2, 3, 4, 5]:
                res = run_single_scenario(s_id, env, args, viewer)
                results.append(res)
            print("\n" + "=" * 80)
            print("ALL 5 SCENARIOS COMPLETED SUCCESSFULLY")
            print("=" * 80)
        else:
            run_single_scenario(args.scenario, env, args, viewer)

    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
