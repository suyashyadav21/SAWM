"""
SAWM - Milestone 1 Demonstration Script
Runs Franka Panda robot simulation with multi-stage movement, contact detection,
joint limit monitoring, and offscreen screenshot capture.

Usage:
  python scripts/run_simulation.py
  python scripts/run_simulation.py --view    # To launch interactive 3D viewer
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.robot import FrankaPandaRobot
from simulation.utils import save_png


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 1 - Franka Panda Simulation Runner")
    parser.add_argument("--config", type=str, default="configs/simulation.yaml", help="Path to config YAML")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D MuJoCo viewer")
    parser.add_argument("--output", type=str, default="results/milestone1_simulation.png", help="Path for screenshot")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier in viewer")
    return parser.parse_args()


def print_telemetry(step_idx: int, obs: dict):
    """Print readable telemetry status line."""
    t = obs["time"]
    robot = obs["robot"]
    ee_pos = robot["ee_position"]
    obj_pos = obs["target_object"]["position"]
    dist = obs["target_object"]["dist_to_ee"]
    contacts = obs["contacts"]
    c_count = contacts["total_contacts"]

    q = robot["joint_positions"]
    q_str = " ".join([f"{val:+.2f}" for val in q])

    status = "SAFE"
    any_violated = any(j["violated"] for j in obs["joint_limits"].values())
    if any_violated or contacts["robot_table_collision"]:
        status = "COLLISION/VIOLATION"

    print(f"[{t:5.2f}s] EE:[{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}] | ObjDist:{dist:.3f}m | Contacts:{c_count} | Status:{status}")


def main():
    args = parse_args()

    print("=" * 70)
    print("SAWM - Franka Panda Robot Simulation (Milestone 1)")
    print("=" * 70)

    # Initialize environment
    print(f"Loading configuration from: {args.config}")
    env = SawmEnvironment(args.config)
    robot = env.robot

    print(f"MuJoCo physics model loaded: {env.config['scene']['model_path']}")
    print(f"Robot DOF: {len(robot.joint_names)} arm joints + gripper")
    print(f"Physics timestep: {env.dt}s (control cycle: {env.dt * env.frame_skip:.3f}s)")
    print("-" * 70)

    # Plan motion stages:
    # Stage 1: Safe transit to Ready pose (above table)
    # Stage 2: Descend to tabletop near target object (demonstrates contact & collision detection)
    # Stage 3: Close gripper (grasp action)
    # Stage 4: Open gripper (release action)
    # Stage 5: Retract to Ready pose (clears table, returns to safe state)
    waypoints = [
        ("Home -> Ready Pose (Safe Free-Space Motion)", env.home_qpos, env.ready_qpos, 60, 0.0),
        ("Ready -> Tabletop Descent (Surface Contact Verification)", env.ready_qpos, env.approach_qpos, 60, 0.0),
        ("Close Gripper (Object Interaction)", env.approach_qpos, env.approach_qpos, 30, 255.0),
        ("Open Gripper (Object Release)", env.approach_qpos, env.approach_qpos, 30, 0.0),
        ("Retract to Ready Pose (Return to Safe Transit)", env.approach_qpos, env.ready_qpos, 60, 0.0),
    ]

    total_steps = sum(w[3] for w in waypoints)
    print(f"Executing multi-stage trajectory ({total_steps} control cycles)...")

    # Optional interactive viewer
    viewer = None
    if args.view:
        import mujoco.viewer
        print("\n[INFO] Launching interactive MuJoCo passive viewer...")
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    current_cycle = 0
    stage_num = 0

    try:
        for stage_name, start_q, target_q, num_steps, gripper_cmd in waypoints:
            stage_num += 1
            print(f"\n--- Stage {stage_num}: {stage_name} ({num_steps} steps) ---")
            traj = FrankaPandaRobot.interpolate_trajectory(start_q, target_q, steps=num_steps, method="cosine")

            for wp in traj:
                current_cycle += 1
                obs = env.step(arm_targets=wp, gripper_target=gripper_cmd)

                # Sync interactive viewer if open
                if viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep((env.dt * env.frame_skip) / max(0.1, args.speed))

                if current_cycle % 15 == 0 or current_cycle == total_steps:
                    print_telemetry(current_cycle, obs)

        # Capture final frame for screenshot
        print("\nCapturing offscreen high-resolution camera view...")
        frame = env.render(camera="workspace_cam", width=1280, height=720)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_png(args.output, frame)
        file_size_kb = os.path.getsize(args.output) / 1024.0
        print(f"Screenshot successfully saved to: {args.output} ({file_size_kb:.1f} KB)")

        # Final verification summary
        final_obs = env.get_observation()
        print("\n" + "=" * 70)
        print("SIMULATION RUN COMPLETED SUCCESSFULLY")
        print(f"Final Sim Time:        {final_obs['time']:.2f} s")
        print(f"Final EE Position:     {np.round(final_obs['robot']['ee_position'], 4)} m")
        print(f"Distance to Target:    {final_obs['target_object']['dist_to_ee']:.4f} m")
        print(f"Active Contacts:       {final_obs['contacts']['total_contacts']}")
        print(f"Robot-Table Collision: {final_obs['contacts']['robot_table_collision']}")
        print(f"All Joint Limits Safe: {robot.is_safe_configuration()}")
        print("=" * 70)

    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
