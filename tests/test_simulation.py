"""
Automated Test Suite for Milestone 1: Minimal Franka Panda Simulation.

Tests:
1. Environment initialization & MuJoCo model verification
2. Franka Panda robot controller & metadata
3. Joint limit checking & validation
4. Smooth trajectory interpolation
5. Robot motion and physics stepping
6. Collision detection & contact querying
7. Multi-camera offscreen rendering & screenshot generation
"""

import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from simulation.robot import FrankaPandaRobot
from simulation.utils import save_png


def test_environment_initialization():
    """Verify environment loads model, creates data, and initializes properly."""
    print("\n--- Test 1: Environment Initialization ---")
    env = SawmEnvironment("configs/simulation.yaml")
    assert env.model is not None, "Model should not be None"
    assert env.data is not None, "Data should not be None"
    assert env.robot is not None, "Robot should not be None"

    print(f"  MuJoCo Version: {mujoco_version()}")
    print(f"  Bodies: {env.model.nbody}, Geoms: {env.model.ngeom}, Actuators: {env.model.nu}")
    print(f"  Timestep: {env.dt}s, Substeps per step: {env.frame_skip}")
    assert env.model.nu == 8, f"Expected 8 actuators, got {env.model.nu}"
    assert env.model.njnt == 10, f"Expected 10 joints (7 arm + 2 finger + 1 object), got {env.model.njnt}"
    print("  [PASS] Environment initialization verified.")
    env.close()


def test_robot_state_and_joint_limits():
    """Verify Franka Panda joint positions, limits, and end-effector pose."""
    print("\n--- Test 2: Robot State & Joint Limits ---")
    env = SawmEnvironment("configs/simulation.yaml")
    robot = env.robot

    # Initial joint positions
    qpos = robot.get_joint_positions()
    print(f"  Initial joint positions (rad): {np.round(qpos, 4)}")
    assert len(qpos) == 7, "Expected 7 arm joint positions"

    # End-effector pose
    ee_pos, ee_quat = robot.get_end_effector_pose()
    print(f"  Initial EE position (m): {np.round(ee_pos, 4)}")
    print(f"  Initial EE quaternion: {np.round(ee_quat, 4)}")
    assert len(ee_pos) == 3, "EE position must be 3D"
    assert ee_pos[2] > 0.2, f"EE height {ee_pos[2]} should be above workstation table"

    # Check joint limits
    limit_checks = robot.check_joint_limits(margin_rad=0.05)
    for jname, info in limit_checks.items():
        print(f"    {jname}: pos={info['position']:+.3f} rad, range=[{info['limits'][0]:+.3f}, {info['limits'][1]:+.3f}], violated={info['violated']}")
        assert not info["violated"], f"Joint {jname} violates limit at initialization!"

    assert robot.is_safe_configuration(), "Robot should start in a safe configuration"
    print("  [PASS] Robot state and joint limits verified.")
    env.close()


def test_trajectory_and_robot_motion():
    """Verify robot responds to joint position commands and tracks trajectory."""
    print("\n--- Test 3: Robot Motion & Trajectory Tracking ---")
    env = SawmEnvironment("configs/simulation.yaml")
    robot = env.robot

    start_q = env.home_qpos.copy()
    target_q = env.ready_qpos.copy()

    # Generate smooth 50-step trajectory
    steps = 50
    trajectory = FrankaPandaRobot.interpolate_trajectory(start_q, target_q, steps=steps, method="cosine")
    assert len(trajectory) == steps, "Trajectory length mismatch"

    initial_ee_pos, _ = robot.get_end_effector_pose()

    # Execute trajectory
    for waypoint in trajectory:
        obs = env.step(arm_targets=waypoint)

    final_q = robot.get_joint_positions()
    final_ee_pos, _ = robot.get_end_effector_pose()

    tracking_error = np.linalg.norm(final_q - target_q)
    ee_displacement = np.linalg.norm(final_ee_pos - initial_ee_pos)

    print(f"  Target joint angles:  {np.round(target_q, 4)}")
    print(f"  Achieved joint angles:{np.round(final_q, 4)}")
    print(f"  Joint tracking error: {tracking_error:.4f} rad")
    print(f"  End-effector displacement: {ee_displacement:.4f} m")

    assert tracking_error < 0.1, f"Tracking error {tracking_error:.4f} exceeds tolerance 0.1 rad"
    assert ee_displacement > 0.05, f"Robot should have visibly moved (displacement={ee_displacement:.4f}m)"
    print("  [PASS] Robot motion and trajectory tracking verified.")
    env.close()


def test_collision_detection():
    """Verify contact and collision querying."""
    print("\n--- Test 4: Collision & Contact Detection ---")
    env = SawmEnvironment("configs/simulation.yaml")

    # Target object should be resting on the table
    summary = env.get_contacts_summary()
    print(f"  Total active contacts in arena: {summary['total_contacts']}")
    print(f"  Object-table contact detected: {summary['object_table_contact']}")
    print(f"  Robot-table collision detected: {summary['robot_table_collision']}")

    assert summary["total_contacts"] > 0, "Expected active contacts (e.g. table on floor, cube on table)"
    assert summary["object_table_contact"], "Target cube should be in contact with the table top"
    assert not summary["robot_table_collision"], "Robot should not collide with table at home pose"

    # Print first few contacts
    contacts = env.get_contacts()
    for i, c in enumerate(contacts[:4]):
        print(f"    Contact {i+1}: {c['body1']}:{c['geom1']} <-> {c['body2']}:{c['geom2']}, dist={c['dist']:.5f}m")

    print("  [PASS] Collision and contact detection verified.")
    env.close()


def test_rendering_and_screenshot():
    """Verify offscreen rendering across cameras and save screenshot."""
    print("\n--- Test 5: Offscreen Rendering & Screenshot ---")
    env = SawmEnvironment("configs/simulation.yaml")

    # Step to ready pose for a clear view
    steps = 30
    trajectory = FrankaPandaRobot.interpolate_trajectory(env.home_qpos, env.ready_qpos, steps=steps)
    for wp in trajectory:
        env.step(arm_targets=wp)

    # Test rendering from workspace_cam
    frame_workspace = env.render(camera="workspace_cam", width=1280, height=720)
    print(f"  Workspace frame shape: {frame_workspace.shape}, dtype: {frame_workspace.dtype}")
    assert frame_workspace.shape == (720, 1280, 3), f"Unexpected shape {frame_workspace.shape}"
    assert frame_workspace.dtype == np.uint8, "Expected uint8 frame"
    assert frame_workspace.mean() > 10.0, "Rendered frame appears to be blank/black"

    # Test front_cam
    frame_front = env.render(camera="front_cam", width=640, height=480)
    assert frame_front.shape == (480, 640, 3)

    # Save milestone screenshot
    os.makedirs("results", exist_ok=True)
    screenshot_path = "results/milestone1_simulation.png"
    save_png(screenshot_path, frame_workspace)
    assert os.path.exists(screenshot_path), "Screenshot file was not created"
    file_size_kb = os.path.getsize(screenshot_path) / 1024.0
    print(f"  Screenshot saved: {screenshot_path} ({file_size_kb:.1f} KB)")
    print("  [PASS] Offscreen rendering and screenshot verified.")
    env.close()


def mujoco_version() -> str:
    import mujoco
    return getattr(mujoco, "__version__", "unknown")


if __name__ == "__main__":
    print("=" * 60)
    print("SAWM - Milestone 1 Automated Test Suite")
    print("=" * 60)
    try:
        test_environment_initialization()
        test_robot_state_and_joint_limits()
        test_trajectory_and_robot_motion()
        test_collision_detection()
        test_rendering_and_screenshot()
        print("\n" + "=" * 60)
        print("ALL MILESTONE 1 TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n[FAIL] Test assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
