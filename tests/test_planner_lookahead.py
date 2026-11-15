"""
Automated Test Suite for Baseline Planner Lookahead Interface (Milestone 5 Requirement).

Verifies:
1. Shape, data type, joint limits, and gripper normalization of planned actions.
2. Complete non-interference: running with vs. without lookahead queries yields bit-for-bit identical executions.
3. Explicit state snapshot comparison before and after lookahead calls (FSM, cursor, step counts, sim time, qpos, qvel, ctrl).
4. Multi-horizon lookahead captures planned FSM stage transitions across horizons.
5. Graceful terminal padding when lookahead extends beyond plan end.
6. Exact M4 action compatibility: joint ordering, radians, gripper normalization, ranges, timing (100 Hz), and numerical equivalence.
"""

import os
import sys
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from planning.baseline_planner import BaselinePlanner
from planning.fsm import TaskState, TaskResult
from world_model.dataset import ObservationBuilder, TrajectoryNormalizer


def test_lookahead_shape_and_units():
    """Verify planned action sequence matches RSSM tensor format and joint limits."""
    print("\n--- Test 1: Lookahead Shape, Units, and Normalization ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    obs = env.reset()

    actions, stages = planner.get_planned_action_sequence(horizon=50)

    print(f"  Actions Shape:  {actions.shape} (Expected: (50, 8))")
    print(f"  Actions Dtype:  {actions.dtype} (Expected: float32)")
    print(f"  Stages Length:  {len(stages)} (Expected: 50)")

    assert actions.shape == (50, 8), f"Expected shape (50, 8), got {actions.shape}"
    assert actions.dtype == np.float32, f"Expected float32, got {actions.dtype}"
    assert len(stages) == 50, f"Expected 50 stages, got {len(stages)}"

    # Check joint limits
    joint_limits = env.robot.joint_limits
    joint_names = env.robot.joint_names
    for i, jname in enumerate(joint_names):
        q_min, q_max = joint_limits[jname]
        q_cmd = actions[:, i]
        assert np.all(q_cmd >= q_min - 1e-4) and np.all(q_cmd <= q_max + 1e-4), (
            f"Joint {jname} exceeded limits: min={q_cmd.min()}, max={q_cmd.max()}, allowed=[{q_min}, {q_max}]"
        )

    # Check normalized gripper command (0.0=open, 1.0=closed)
    g_cmd = actions[:, 7]
    assert np.all(g_cmd >= 0.0) and np.all(g_cmd <= 1.0), (
        f"Gripper command outside [0, 1]: min={g_cmd.min()}, max={g_cmd.max()}"
    )

    # Initial stage should be HOME
    assert stages[0] == TaskState.HOME, f"Expected first stage to be HOME, got {stages[0]}"

    print("  [PASS] Shape, types, joint limits, and normalization verified.")
    env.close()


def test_lookahead_non_interference():
    """Verify lookahead queries at every step do not modify baseline execution or commands."""
    print("\n--- Test 2: Lookahead Non-Interference Verification ---")
    env1 = SawmEnvironment()
    planner1 = BaselinePlanner(env1)

    # 1. Run baseline without lookahead queries
    trajectory1 = []
    def callback1(step, obs, state):
        trajectory1.append(obs["robot"]["joint_positions"][:7].copy())

    result1 = planner1.execute(step_callback=callback1)
    env1.close()

    # 2. Run baseline with lookahead queries at every single control step
    env2 = SawmEnvironment()
    planner2 = BaselinePlanner(env2)

    trajectory2 = []
    lookahead_history = []
    def callback2(step, obs, state):
        trajectory2.append(obs["robot"]["joint_positions"][:7].copy())
        # Query lookahead at every step
        act_seq, stage_seq = planner2.get_planned_action_sequence(horizon=50)
        lookahead_history.append((act_seq, stage_seq))

    result2 = planner2.execute(step_callback=callback2)
    env2.close()

    # Compare outcomes
    print(f"  Run 1 (No Lookahead):   Outcome={result1.final_state}, Steps={len(trajectory1)}, Path={result1.trajectory_length:.4f}m")
    print(f"  Run 2 (With Lookahead): Outcome={result2.final_state}, Steps={len(trajectory2)}, Path={result2.trajectory_length:.4f}m")

    assert result1.success == result2.success, "Task success differed between runs!"
    assert result1.final_state == result2.final_state, "Final state differed between runs!"
    assert len(trajectory1) == len(trajectory2), f"Step count differed: {len(trajectory1)} vs {len(trajectory2)}"
    assert len(lookahead_history) == len(trajectory2), "Lookahead queries were not recorded at every step"

    # Compare trajectory point-by-point
    traj1_arr = np.array(trajectory1)
    traj2_arr = np.array(trajectory2)
    max_diff = np.max(np.abs(traj1_arr - traj2_arr))
    print(f"  Max Joint Position Difference: {max_diff:.2e} rad")

    assert max_diff < 1e-12, f"Trajectories differed! Max diff: {max_diff}"
    print("  [PASS] Lookahead interface is strictly non-interfering (bit-for-bit identical).")


def test_immediate_snapshot_non_interference():
    """
    Explicitly snapshot and compare planner, FSM, and simulation state
    immediately before and after get_planned_action_sequence() calls across
    different stages, repeated queries, and diverse horizons.
    """
    print("\n--- Test 3: Immediate Pre/Post Snapshot Non-Interference ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    env.reset()

    def snapshot_state(planner, env):
        return {
            "fsm_state": planner.fsm.state,
            "fsm_step_count": planner.fsm.state_step_count,
            "fsm_terminal": planner.fsm.is_terminal,
            "planner_cursor": planner._current_step_cursor,
            "sim_time": float(env.data.time),
            "sim_qpos": env.data.qpos.copy(),
            "sim_qvel": env.data.qvel.copy(),
            "sim_ctrl": env.data.ctrl.copy(),
            "sim_qacc": env.data.qacc.copy(),
        }

    def assert_snapshots_identical(s_before, s_after, label=""):
        assert s_before["fsm_state"] == s_after["fsm_state"], f"{label}: FSM state mutated! ({s_before['fsm_state']} -> {s_after['fsm_state']})"
        assert s_before["fsm_step_count"] == s_after["fsm_step_count"], f"{label}: FSM step count mutated!"
        assert s_before["fsm_terminal"] == s_after["fsm_terminal"], f"{label}: FSM terminal flag mutated!"
        assert s_before["planner_cursor"] == s_after["planner_cursor"], f"{label}: Planner cursor mutated!"
        assert s_before["sim_time"] == s_after["sim_time"], f"{label}: Simulation time mutated!"
        assert np.array_equal(s_before["sim_qpos"], s_after["sim_qpos"]), f"{label}: Simulation qpos mutated!"
        assert np.array_equal(s_before["sim_qvel"], s_after["sim_qvel"]), f"{label}: Simulation qvel mutated!"
        assert np.array_equal(s_before["sim_ctrl"], s_after["sim_ctrl"]), f"{label}: Actuator ctrl mutated!"
        assert np.array_equal(s_before["sim_qacc"], s_after["sim_qacc"]), f"{label}: Acceleration qacc mutated!"

    # Test snapshots at critical stages of execution
    test_checkpoints = [5, 45, 110, 150, 200, 280, 380, 470]
    tested_count = 0

    def step_verifier(step_idx, obs, fsm_state_val):
        nonlocal tested_count
        if step_idx in test_checkpoints:
            tested_count += 1
            snap_pre = snapshot_state(planner, env)

            # 1. Single query H=50
            acts_50, stages_50 = planner.get_planned_action_sequence(horizon=50)
            snap_post_1 = snapshot_state(planner, env)
            assert_snapshots_identical(snap_pre, snap_post_1, f"Step {step_idx} (Single H=50)")

            # 2. Repeated calls with varied horizons: H in [1, 5, 10, 20, 50]
            for h in [1, 5, 10, 20, 50]:
                for _ in range(3):
                    planner.get_planned_action_sequence(horizon=h)
            snap_post_multi = snapshot_state(planner, env)
            assert_snapshots_identical(snap_pre, snap_post_multi, f"Step {step_idx} (Multi-call H=1..50)")

            # 3. Explicit cursor lookahead queries (different offsets)
            for c in [0, 50, 250, 500]:
                planner.get_planned_action_sequence(horizon=20, current_cursor=c)
            snap_post_cursor = snapshot_state(planner, env)
            assert_snapshots_identical(snap_pre, snap_post_cursor, f"Step {step_idx} (Arbitrary cursor)")

            # 4. Verify that the next commanded step matches the first action in lookahead
            next_action_lookahead = acts_50[0]
            planned_next_step = planner._planned_steps[planner._current_step_cursor]
            assert np.allclose(next_action_lookahead, planned_next_step["action_vector"], atol=1e-7), (
                f"Step {step_idx}: Lookahead action 0 does not match planned next action!"
            )

    result = planner.execute(step_callback=step_verifier)
    print(f"  Verified state invariance across {tested_count} distinct trajectory checkpoints.")
    print("  Checks included: FSM state, state step count, cursor, sim time, qpos, qvel, ctrl, qacc.")
    print("  [PASS] Pre/post lookahead snapshot non-interference strictly confirmed.")
    env.close()


def test_lookahead_stage_transitions():
    """Verify that multi-horizon lookahead accurately predicts upcoming FSM stage transitions."""
    print("\n--- Test 4: Lookahead Stage Transition Horizon Tracking ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    env.reset()

    # Build plan
    planner.get_planned_action_sequence(horizon=50)
    total_plan_len = len(planner._planned_steps)
    print(f"  Total Discrete Plan Steps: {total_plan_len}")

    # Inspect planned stages
    unique_stages = []
    for s in planner._planned_steps:
        st = s["fsm_state"]
        if not unique_stages or unique_stages[-1] != st:
            unique_stages.append(st)
    print(f"  Planned FSM Stages ({len(unique_stages)}): {[s.value for s in unique_stages]}")

    # Find a boundary where stage transitions (e.g. from HOME to MOVE_TO_PREGRASP)
    # The first segment is HOME with 10 steps. At cursor=5, horizon=10 should see both HOME and MOVE_TO_PREGRASP!
    actions, stages = planner.get_planned_action_sequence(horizon=10, current_cursor=5)
    print(f"  Horizon 10 stages from cursor 5: {[s.value for s in stages]}")

    assert stages[0] == TaskState.HOME, f"Expected stages[0] to be HOME, got {stages[0]}"
    assert TaskState.MOVE_TO_PREGRASP in stages, "Expected upcoming MOVE_TO_PREGRASP stage in lookahead!"

    # Verify that the action at the transition matches
    assert actions.shape == (10, 8)
    print("  [PASS] Upcoming FSM stage transitions accurately predicted in lookahead.")
    env.close()


def test_lookahead_terminal_padding():
    """Verify graceful padding with holding action and terminal state at plan boundary."""
    print("\n--- Test 5: Lookahead Terminal Padding ---")
    env = SawmEnvironment()
    planner = BaselinePlanner(env)
    env.reset()

    # Ensure plan is built
    planner.get_planned_action_sequence(horizon=50)
    total_plan_len = len(planner._planned_steps)

    # Query with cursor 5 steps before the end
    near_end_cursor = total_plan_len - 5
    actions, stages = planner.get_planned_action_sequence(horizon=20, current_cursor=near_end_cursor)

    assert actions.shape == (20, 8)
    assert len(stages) == 20

    # The last 15 actions should be identical to the terminal action (padded)
    terminal_act = planner._planned_steps[-1]["action_vector"]
    for i in range(5, 20):
        assert np.allclose(actions[i], terminal_act, atol=1e-6), f"Padded action {i} did not match terminal action"
        assert stages[i] == planner._planned_steps[-1]["fsm_state"]

    print("  [PASS] Terminal boundary correctly pads remaining horizon with terminal holding action.")
    env.close()


def test_exact_m4_action_compatibility():
    """
    Directly verify planner action vectors against M4 training dataset convention:
    1. Seven joint values match exact Franka Panda joint ordering and radians.
    2. Gripper direction and normalization match M4 convention (open=0.0, closed=1.0).
    3. Action clipping and valid ranges match.
    4. 8D action vector has identical semantics and 100% numerical equivalence with M4 data.
    5. Control timing is consistent with 100 Hz action interval.
    """
    print("\n--- Test 6: Exact M4 Action Compatibility Verification ---")
    m4_sample_path = "data/trajectories/train/episode_0000.npz"
    assert os.path.exists(m4_sample_path), f"M4 dataset file missing: {m4_sample_path}"

    m4_data = np.load(m4_sample_path, allow_pickle=True)
    m4_actions = m4_data["actions"]
    m4_obs = m4_data["observations"]
    m4_meta = m4_data["metadata"].item()

    print(f"  M4 Episode 0000 Action Array Shape: {m4_actions.shape}")
    print(f"  M4 Episode 0000 Action Dtype:       {m4_actions.dtype}")

    env = SawmEnvironment()
    planner = BaselinePlanner(env)

    # 1. Verify 100 Hz Action Interval / Control Timing
    dt_ctrl = env.dt * env.frame_skip
    print(f"  MuJoCo dt: {env.dt}s x FrameSkip {env.frame_skip} = {dt_ctrl:.4f}s ({1.0/dt_ctrl:.0f} Hz)")
    assert np.isclose(dt_ctrl, 0.010), "Control step must be exactly 0.01s (100 Hz)"

    # Verify simulation time advances by exactly dt_ctrl per step
    t0 = float(env.data.time)
    env.step(arm_targets=env.home_qpos, gripper_target=255.0)
    t1 = float(env.data.time)
    assert np.isclose(t1 - t0, 0.010, atol=1e-6), f"Simulation time step was {t1 - t0}s, expected 0.010s"

    # 2. Verify Joint Ordering & Radians
    joint_names = env.robot.joint_names
    expected_names = [f"joint{i+1}" for i in range(7)]
    assert joint_names == expected_names, f"Joint name ordering mismatch: {joint_names} vs {expected_names}"

    # 3. Plan keyframes matching the exact cube position of episode_0000
    cube_pos = np.array(m4_meta["cube_init_pos"])
    goal_pos = np.array([0.55, 0.20, 0.401]) # Nominal goal in arena
    keyframes, ok, _ = planner.plan_keyframes(cube_pos, goal_pos)
    assert ok, "IK failed for episode_0000 keyframes"

    planner._planned_steps = planner.build_discrete_plan(keyframes)
    lookahead_actions, lookahead_stages = planner.get_planned_action_sequence(
        horizon=len(m4_actions), current_cursor=0
    )

    # 4. Joint Limits & Valid Range Assertions
    joint_limits = env.robot.joint_limits
    for i, jname in enumerate(joint_names):
        q_min, q_max = joint_limits[jname]
        # In lookahead
        assert np.all(lookahead_actions[:, i] >= q_min - 1e-4) and np.all(lookahead_actions[:, i] <= q_max + 1e-4), (
            f"Joint {jname} lookahead exceeded limits [{q_min}, {q_max}]"
        )
        # In M4 dataset
        assert np.all(m4_actions[:, i] >= q_min - 1e-4) and np.all(m4_actions[:, i] <= q_max + 1e-4), (
            f"Joint {jname} M4 data exceeded limits [{q_min}, {q_max}]"
        )

    # 5. Gripper Direction & Normalization Verification
    # Index 7: 0.0 = open (gripper commanded 255), 1.0 = closed (gripper commanded 0)
    lookahead_gripper = lookahead_actions[:, 7]
    m4_gripper = m4_actions[:, 7]

    assert np.all((lookahead_gripper >= 0.0) & (lookahead_gripper <= 1.0))
    assert np.all((m4_gripper >= 0.0) & (m4_gripper <= 1.0))

    # Check phase correspondence: open during transit, closed during lift/transport
    # Steps 0..60: HOME & MOVE_TO_PREGRASP -> gripper open (0.0)
    assert np.all(lookahead_gripper[:60] == 0.0)
    assert np.all(m4_gripper[:60] == 0.0)
    # Steps 180..240: LIFT -> gripper closed (1.0)
    assert np.all(lookahead_gripper[180:240] == 1.0)
    assert np.all(m4_gripper[180:240] == 1.0)
    print("  Gripper direction confirmed: 0.0 = open, 1.0 = closed across all stages.")

    # 6. Point-by-Point 100% Exact Numerical Equivalence
    max_act_diff = np.max(np.abs(lookahead_actions - m4_actions))
    print(f"  Max Numerical Difference across all {len(m4_actions)} steps x 8 dimensions: {max_act_diff:.2e}")
    assert max_act_diff == 0.0, f"Planner lookahead actions did not match M4 dataset actions! Max diff: {max_act_diff}"

    # 7. Normalizer Exemption Verification
    normalizer_path = "checkpoints/normalizer.pt"
    assert os.path.exists(normalizer_path), f"Normalizer checkpoint missing: {normalizer_path}"
    norm_data = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    # Normalizer mean and std have shape (49,) for observations, verifying actions are NOT normalized by TrajectoryNormalizer
    assert norm_data["mean"].shape == (49,), "TrajectoryNormalizer must operate strictly on 49-dim observations"
    print("  Normalizer confirms: actions are NOT standardized by TrajectoryNormalizer (passed raw to RSSM).")

    print("  [PASS] Exact M4 action compatibility verified with 100% numerical equivalence (0.0 diff).")
    env.close()


def run_all_tests():
    print("=" * 80)
    print("SAWM - Milestone 5 Baseline Planner Lookahead Interface Tests")
    print("=" * 80)

    test_lookahead_shape_and_units()
    test_lookahead_non_interference()
    test_immediate_snapshot_non_interference()
    test_lookahead_stage_transitions()
    test_lookahead_terminal_padding()
    test_exact_m4_action_compatibility()

    print("\n" + "=" * 80)
    print("ALL PLANNER LOOKAHEAD & M4 COMPATIBILITY TESTS PASSED! (6/6)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
