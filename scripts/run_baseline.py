"""
SAWM - Milestone 3 Baseline Task Planner Runner
Executes conventional, unassisted pick-and-place task using:
- 6D Pose Inverse Kinematics (DLS with posture optimization)
- 8-stage Finite-State Machine (FSM)
- Strict baseline independence (passive safety monitor)
- Offscreen visualization and interactive 3D viewer support

Usage:
  python scripts/run_baseline.py
  python scripts/run_baseline.py --view    # To launch interactive 3D viewer
  python scripts/run_baseline.py --output results/milestone3_baseline.png
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.utils import save_png
from planning.baseline_planner import BaselinePlanner


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 3 - Baseline Task Planner Runner")
    parser.add_argument("--config", type=str, default="configs/planner.yaml", help="Path to planner YAML config")
    parser.add_argument("--sim-config", type=str, default="configs/simulation.yaml", help="Path to simulation YAML config")
    parser.add_argument("--safety-config", type=str, default="configs/safety.yaml", help="Path to safety YAML config")
    parser.add_argument("--view", action="store_true", help="Launch interactive 3D MuJoCo viewer")
    parser.add_argument("--output", type=str, default="results/milestone3_baseline.png", help="Path for screenshot")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier in viewer")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SAWM - Franka Panda Baseline Task Planner (Milestone 3)")
    print("=" * 80)
    print("Task: Conventional Pick-and-Place (FSM + 6D IK)")
    print("Safety Mode: PASSIVE (Zero intervention / Independent Baseline)")
    print(f"Planner Config: {args.config}")
    print(f"Interactive Viewer: {'ENABLED' if args.view else 'DISABLED (Offscreen only)'}")
    print("=" * 80)

    # Initialize environment and planner
    env = SawmEnvironment(config_path=args.sim_config, safety_config_path=args.safety_config)
    planner = BaselinePlanner(env, config_path=args.config)

    # Optional interactive viewer
    viewer = None
    if args.view:
        import mujoco.viewer
        print("\n[INFO] Launching interactive MuJoCo passive viewer...")
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    last_printed_state = None

    def step_callback(step_idx: int, obs: dict, current_state: str):
        nonlocal last_printed_state

        # Sync interactive viewer
        if viewer is not None and viewer.is_running():
            viewer.sync()
            time.sleep((env.dt * env.frame_skip) / max(0.1, args.speed))

        # Print telemetry on state transition or periodically
        is_transition = current_state != last_printed_state
        if is_transition or step_idx % 25 == 0:
            t = obs["time"]
            ee = obs["robot"]["ee_position"]
            obj = obs["target_object"]["position"]
            contact = obs["contacts"]["robot_object_contact"]
            safety_state = obs.get("safety", {}).state if hasattr(obs.get("safety", {}), "state") else "UNKNOWN"
            grip_w = obs["robot"]["gripper_position"]

            prefix = ">>> STATE TRANSITION -> " if is_transition else "    "
            print(
                f"{prefix}[{t:5.2f}s | step {step_idx:4d}] "
                f"State: {current_state:<18s} | "
                f"EE: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}] | "
                f"Obj: [{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}] | "
                f"Grasp: {'YES' if contact else 'NO '} (w={grip_w*1000:4.1f}mm) | "
                f"Safety: {safety_state}"
            )
            last_printed_state = current_state

    try:
        print("\n[INFO] Commencing baseline pick-and-place execution...")
        result = planner.execute(step_callback=step_callback)

        # Capture final frame for offscreen screenshot
        print(f"\n[INFO] Capturing high-resolution camera view from 'workspace_cam'...")
        frame = env.render(camera="workspace_cam", width=1280, height=720)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_png(args.output, frame)
        file_size_kb = os.path.getsize(args.output) / 1024.0
        print(f"[INFO] Screenshot successfully saved to: {args.output} ({file_size_kb:.1f} KB)")

        # Print summary report
        print("\n" + "=" * 80)
        print("BASELINE TASK EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Task Outcome:             {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Final State:              {result.final_state}")
        if result.failure_reason:
            print(f"Failure Reason:           {result.failure_reason}")
        print(f"Execution Sim Time:       {result.execution_time:.3f} s")
        print(f"Trajectory Path Length:   {result.trajectory_length:.3f} m")
        print(f"Grasp Verified:           {'YES' if result.grasp_success else 'NO'}")
        print(f"Final Placement Error:    {result.placement_error*1000:.2f} mm")
        print(f"Joint Limit Violations:   {result.joint_limit_violations}")
        print(f"State Sequence:           {' -> '.join(result.state_history)}")
        print(f"Stages Completed:         {len(result.stages_completed)} stages: {result.stages_completed}")
        
        # Safety summary
        ss = result.safety_summary
        print("-" * 80)
        print("PASSIVE SAFETY TELEMETRY (BASELINE BENCHMARK):")
        if ss:
            states = ss.get("safety_states", {})
            print(f"  Steps in SAFE:          {states.get('SAFE', 0)}")
            print(f"  Steps in WARNING:       {states.get('WARNING', 0)}")
            print(f"  Steps in DANGER:        {states.get('DANGER', 0)}")
            min_c = ss.get("min_clearance")
            print(f"  Min Hazard Clearance:   {f'{min_c*1000:.1f} mm' if min_c is not None else 'N/A (No obstacles active)'}")
            print(f"  Collisions Observed:    {ss.get('collisions', 0)}")
            print(f"  Near Misses:            {ss.get('near_misses', 0)}")
        print("=" * 80)

        return 0 if result.success else 1

    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    sys.exit(main())
