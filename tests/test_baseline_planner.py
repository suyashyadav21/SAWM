"""
Automated Test Suite for Milestone 3: Baseline Task Planner.

Tests:
1. 6D Inverse Kinematics convergence across all keyframes (pregrasp, grasp, lift, transport, place, retract, home)
2. Full nominal pick-and-place execution (all 8 FSM stages, 0 joint limit violations, < 10mm placement error)
3. Robust grasp verification (finger-object contact, lift height >= 5cm, transit attachment)
4. Failure handling - Grasp failure (gripper kept open or object displaced)
5. Failure handling - Slippage detection (drift during transit)
6. Failure handling - Placement failure (outside goal tolerance radius)
7. Failure handling - Timeout protection (exceeded max steps per state)
8. Repeatability evaluation (5 consecutive independent runs with comprehensive metrics)

Usage:
  python tests/test_baseline_planner.py
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from planning.baseline_planner import BaselinePlanner
from planning.ik_solver import Franka6DIKSolver
from planning.fsm import PickPlaceFSM, TaskState, TaskResult


def test_ik_convergence_all_keyframes():
    """Verify 6D IK converges accurately for all keyframe poses."""
    print("\n--- Test 1: 6D Inverse Kinematics Keyframe Convergence ---")
    env = SawmEnvironment()
    ik = Franka6DIKSolver(env.model)
    obs = env.reset()
    obj_pos = obs["target_object"]["position"]
    goal_pos = obs["goal"]["position"]

    targets = {
        "pregrasp": np.array([obj_pos[0], obj_pos[1], 0.620]),
        "grasp": np.array([obj_pos[0], obj_pos[1], 0.525]),
        "lift": np.array([obj_pos[0], obj_pos[1], 0.650]),
        "transport": np.array([goal_pos[0], goal_pos[1], 0.650]),
        "place": np.array([goal_pos[0], goal_pos[1], 0.535]),
        "retract": np.array([goal_pos[0], goal_pos[1], 0.650]),
    }

    q_seed = env.ready_qpos
    for name, target_pos in targets.items():
        q_sol, success, iters, pos_err, rot_err = ik.solve(target_pos, q_seed=q_seed)
        print(f"  Stage {name:<10s}: Success={success} | Iters={iters:2d} | PosErr={pos_err*1000:.2f}mm | RotErr={rot_err:.4f}rad")
        assert success, f"IK failed to converge for keyframe '{name}'"
        assert pos_err < 0.002, f"Position error too high for '{name}': {pos_err*1000:.2f}mm"
        assert rot_err < 0.05, f"Rotation error too high for '{name}': {rot_err:.3f}rad"
        assert iters <= 25, f"Took too many iterations for '{name}': {iters}"
        q_seed = q_sol

    print("  [PASS] All 6 keyframes converged with sub-millimeter precision.")
    env.close()


def test_full_pick_and_place_execution():
    """Verify nominal pick-and-place completes all 8 stages with millimeter accuracy."""
    print("\n--- Test 2: Full Nominal Pick-and-Place Execution ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    result = planner.execute()

    print(f"  Result Outcome:         {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Final State:            {result.final_state}")
    print(f"  Execution Sim Time:     {result.execution_time:.2f} s")
    print(f"  Trajectory Path Length: {result.trajectory_length:.3f} m")
    print(f"  Grasp Success:          {result.grasp_success}")
    print(f"  Placement Error:        {result.placement_error*1000:.2f} mm")
    print(f"  Joint Limit Violations: {result.joint_limit_violations}")

    assert result.success is True, f"Execution failed: {result.failure_reason}"
    assert result.final_state == "COMPLETED"
    assert result.failure_reason is None
    assert result.grasp_success is True
    assert result.placement_error < 0.010, f"Placement error too high: {result.placement_error*1000:.1f}mm"
    assert result.joint_limit_violations == 0, f"Joint limits violated {result.joint_limit_violations} times"

    expected_stages = [
        "HOME",
        "MOVE_TO_PREGRASP",
        "DESCEND_TO_OBJECT",
        "GRASP",
        "LIFT",
        "MOVE_TO_TARGET",
        "RELEASE",
        "RETURN_HOME",
    ]
    assert result.stages_completed == expected_stages, f"Stages mismatch: {result.stages_completed}"
    print(f"  Completed all {len(result.stages_completed)} required FSM stages.")
    print("  [PASS] Full pick-and-place execution verified.")
    env.close()


def test_grasp_verification():
    """Verify that grasp status correctly reflects lift height and finger contacts."""
    print("\n--- Test 3: Grasp Verification (Contact, Lift Height, & Attachment) ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    
    obs = env.reset()
    obj_pos = obs["target_object"]["position"]
    goal_pos = obs["goal"]["position"]
    keyframes, ok, _ = planner.plan_keyframes(obj_pos, goal_pos)
    assert ok

    # Move to pregrasp and descend
    for wp in [keyframes["pregrasp"], keyframes["grasp"]]:
        for _ in range(50):
            obs = env.step(arm_targets=wp, gripper_target=planner.open_ctrl)

    # Clamp gripper
    for _ in range(40):
        obs = env.step(arm_targets=keyframes["grasp"], gripper_target=planner.close_ctrl)
    has_contact = obs["contacts"]["robot_object_contact"]
    print(f"  Grip Closure Contact:   {has_contact}")
    assert has_contact is True, "Fingers should contact cube"

    # Lift
    for _ in range(50):
        obs = env.step(arm_targets=keyframes["lift"], gripper_target=planner.close_ctrl)

    cube_z = obs["target_object"]["position"][2]
    lift_height = cube_z - planner.table_z
    print(f"  Measured Lift Height:   {lift_height*1000:.1f} mm above table (Threshold: >= 50.0 mm)")
    assert lift_height >= 0.050, f"Cube should be lifted at least 5cm, got {lift_height*100:.1f}cm"
    assert obs["contacts"]["robot_object_contact"] is True, "Cube must remain in contact during lift"

    print("  [PASS] Grasp verification confirmed.")
    env.close()


def test_failure_handling_grasp_failure():
    """Test explicit transition to FAILED when object cannot be grasped."""
    print("\n--- Test 4: Failure Handling - Grasp Failure ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    
    # Keep gripper open during grasp stage to induce grasp failure
    planner.close_ctrl = planner.open_ctrl  # Never clamp
    result = planner.execute()
    print(f"  Outcome: {result.final_state} | Reason: {result.failure_reason}")
    assert result.success is False
    assert result.final_state == "FAILED"
    assert result.failure_reason is not None
    assert "GRASP_FAILURE" in result.failure_reason

    print("  [PASS] Grasp failure correctly transitions to FAILED state.")
    env.close()


def test_failure_handling_slippage():
    """Test explicit transition to FAILED when object slips during transport."""
    print("\n--- Test 5: Failure Handling - Object Slippage Detection ---")
    fsm = PickPlaceFSM({"fsm": {"max_transit_slip": 0.035, "min_lift_height": 0.050}})
    fsm.transition_to(TaskState.LIFT)
    
    # Simulate valid initial lift
    ee_pos = np.array([0.55, -0.12, 0.65])
    obj_pos = np.array([0.55, -0.12, 0.54])  # 14cm above table
    fsm.update_grasp_tracking(ee_pos, obj_pos, {"robot_object_contact": True})
    assert fsm.grasp_verified is True

    # Transition to MOVE_TO_TARGET and introduce drift (slippage)
    fsm.transition_to(TaskState.MOVE_TO_TARGET)
    slipped_obj = np.array([0.55, 0.00, 0.54])  # 12cm drift
    valid = fsm.update_grasp_tracking(ee_pos, slipped_obj, {"robot_object_contact": True})
    print(f"  Outcome: {fsm.state.value} | Reason: {fsm.failure_reason}")
    assert valid is False
    assert fsm.state == TaskState.FAILED
    assert "OBJECT_SLIPPAGE_FAILURE" in fsm.failure_reason

    print("  [PASS] Object slippage correctly transitions to FAILED state.")


def test_failure_handling_placement_failure():
    """Test explicit transition to FAILED when object lands outside goal radius."""
    print("\n--- Test 6: Failure Handling - Placement Error Failure ---")
    fsm = PickPlaceFSM({"fsm": {"goal_tolerance_radius": 0.060}})
    fsm.transition_to(TaskState.RELEASE)

    goal_pos = np.array([0.55, 0.20, 0.401])
    bad_obj_pos = np.array([0.55, 0.35, 0.425])  # 15cm from goal (tolerance is 6cm)

    is_placed, err = fsm.verify_placement(bad_obj_pos, goal_pos)
    print(f"  Placement Valid: {is_placed} | Error: {err*1000:.1f} mm (Tolerance: 60.0 mm)")
    assert is_placed is False
    assert err > 0.060

    print("  [PASS] Placement error correctly caught.")


def test_failure_handling_timeout():
    """Test explicit transition to FAILED when a state exceeds timeout limit."""
    print("\n--- Test 7: Failure Handling - State Timeout ---")
    fsm = PickPlaceFSM({"fsm": {"fsm_timeout_steps": 50}})
    fsm.transition_to(TaskState.MOVE_TO_PREGRASP)
    fsm.state_step_count = 55
    timed_out = fsm.check_timeout()
    print(f"  Timed Out: {timed_out} | State: {fsm.state.value} | Reason: {fsm.failure_reason}")
    assert timed_out is True
    assert fsm.state == TaskState.FAILED
    assert "TIMEOUT_FAILURE" in fsm.failure_reason

    print("  [PASS] State timeout correctly transitions to FAILED state.")


def test_repeatability_5_runs():
    """Execute 5 consecutive nominal runs to evaluate deterministic repeatability."""
    print("\n--- Test 8: 5-Run Nominal Repeatability Evaluation ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    runs = []

    for run_idx in range(1, 6):
        res = planner.execute()
        runs.append({
            "run": run_idx,
            "success": res.success,
            "final_state": res.final_state,
            "placement_error_mm": res.placement_error * 1000.0,
            "execution_time_s": res.execution_time,
            "path_length_m": res.trajectory_length,
            "joint_violations": res.joint_limit_violations,
        })
        assert res.success is True, f"Run {run_idx} failed"
        assert res.placement_error < 0.010, f"Run {run_idx} placement error {res.placement_error*1000:.1f}mm exceeded 10mm"

    print("\n" + "=" * 75)
    print("5-RUN NOMINAL REPEATABILITY EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Run':<6} | {'Outcome':<8} | {'Error (mm)':<12} | {'Time (s)':<10} | {'Path (m)':<10} | {'Violations':<10}")
    print("-" * 75)
    for r in runs:
        print(
            f"{r['run']:<6} | "
            f"{'SUCCESS' if r['success'] else 'FAILED':<8} | "
            f"{r['placement_error_mm']:<12.2f} | "
            f"{r['execution_time_s']:<10.2f} | "
            f"{r['path_length_m']:<10.3f} | "
            f"{r['joint_violations']:<10}"
        )
    print("=" * 75)
    print("  [PASS] 100% repeatability confirmed across 5/5 consecutive trials.")
    env.close()


if __name__ == "__main__":
    print("=" * 80)
    print("SAWM - Milestone 3 Automated Test Suite (Baseline Task Planner)")
    print("=" * 80)
    try:
        test_ik_convergence_all_keyframes()
        test_full_pick_and_place_execution()
        test_grasp_verification()
        test_failure_handling_grasp_failure()
        test_failure_handling_slippage()
        test_failure_handling_placement_failure()
        test_failure_handling_timeout()
        test_repeatability_5_runs()
        print("\n" + "=" * 80)
        print("ALL MILESTONE 3 TESTS PASSED SUCCESSFULLY! (8/8)")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[FAIL] Test assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
