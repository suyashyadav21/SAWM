"""
Automated Test Suite for Milestone 6 Step 2 MVP Predictive Intervention.

Verifies:
1. Fresh H10 telemetry processing and normal speed maintenance.
2. Stale and expired H10 telemetry fail-safe policy.
3. Invalid, NaN, infinite, negative-age, and overflowed telemetry rejection.
4. Moderate risk predictive slowdown escalation (P10 >= 0.36 or d10 <= 0.08m).
5. Critical fast-path predictive intervention (P10 >= 0.70 or d10 <= 0.03m).
6. Layer 0 reactive override priority (physical collision unconditionally overrides predictive state).
7. Actuator inhibition recovery protection (no autonomous resumption from latched emergency).
8. H10Predictor online state tracking and forward rollout sanity.
9. Scenario A demonstration verification (nominal pick-and-place completes 100%).
10. Scenario B demonstration verification (predictive slowdown/arrest avoids physical contact).
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from planning.baseline_planner import BaselinePlanner
from safety.safety_supervisor import (
    SafetySupervisor,
    SupervisorState,
    StopMode,
    ReactiveTelemetry,
    TacticalTelemetry,
    StrategicTelemetry,
)
from safety.predictive_intervention import H10Predictor
from scripts.run_predictive_demo import run_scenario_a_normal, run_scenario_b_predictive_hazard


@pytest.fixture(scope="module")
def shared_env():
    env = SawmEnvironment("configs/simulation.yaml")
    yield env
    env.close()


# =============================================================================
# 1. Fresh H10 Telemetry Processing
# =============================================================================
def test_fresh_h10_telemetry():
    """Verify that fresh low-risk H10 telemetry maintains NORMAL state at 1.0x speed."""
    supervisor = SafetySupervisor(config={"require_h50": False, "h10_fresh_max_ms": 20.0, "h10_stale_max_ms": 100.0})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.35)

    # Fresh low-risk telemetry (age = 10 ms <= 20 ms)
    tac_fresh = TacticalTelemetry(
        p10=0.04,
        d10_clear=0.35,
        c_harm=0.0,
        timestamp=0.10,
        age_ms=10.0,
        human_proxy_distance=1.5,
    )
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_fresh, current_time=0.11)
    assert cmd.state == SupervisorState.NORMAL
    assert cmd.speed_scale == 1.0
    assert cmd.is_latched is False


# =============================================================================
# 2. Stale and Expired H10 Telemetry
# =============================================================================
def test_stale_and_expired_h10_telemetry():
    """Verify that stale H10 reduces speed and expired H10 forces SLOWDOWN."""
    supervisor = SafetySupervisor(config={"require_h50": False, "h10_fresh_max_ms": 20.0, "h10_stale_max_ms": 100.0})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.35)

    # Stale H10: age = 50 ms (between 20 ms and 100 ms) -> CAUTION with speed <= 0.50x
    tac_stale = TacticalTelemetry(p10=0.04, d10_clear=0.35, timestamp=0.10, age_ms=50.0)
    cmd_stale = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_stale, current_time=0.15)
    assert cmd_stale.state == SupervisorState.CAUTION
    assert cmd_stale.speed_scale <= 0.50

    # Expired H10: age = 120 ms (> 100 ms) -> SLOWDOWN
    supervisor_fresh = SafetySupervisor(config={"require_h50": False, "h10_fresh_max_ms": 20.0, "h10_stale_max_ms": 100.0})
    tac_expired = TacticalTelemetry(p10=0.04, d10_clear=0.35, timestamp=0.10, age_ms=120.0)
    cmd_expired = supervisor_fresh.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_expired, current_time=0.22)
    assert cmd_expired.state == SupervisorState.SLOWDOWN
    assert any("EXPIRED" in a.upper() for a in cmd_expired.active_alarms)


# =============================================================================
# 3. Invalid Telemetry Rejection
# =============================================================================
def test_invalid_telemetry_rejection():
    """Verify rejection of NaN, infinite, negative-age, overflowed, and missing telemetry."""
    supervisor = SafetySupervisor(config={"require_h50": False})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.zeros(7)
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.35)

    # 1. Missing tactical (None)
    cmd_none = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=None, current_time=0.1)
    assert cmd_none.state == SupervisorState.SLOWDOWN
    assert any("MISSING" in a.upper() for a in cmd_none.active_alarms)

    # 2. NaN in timestamp
    tac_nan_ts = TacticalTelemetry(p10=0.05, d10_clear=0.30, timestamp=float("nan"))
    cmd_nan_ts = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_nan_ts, current_time=0.1)
    assert cmd_nan_ts.state in [SupervisorState.SLOWDOWN, SupervisorState.EMERGENCY_STOP]
    assert cmd_nan_ts.state != SupervisorState.NORMAL

    # 3. Infinite current_time
    tac_ok = TacticalTelemetry(p10=0.05, d10_clear=0.30, timestamp=0.10)
    cmd_inf_t = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_ok, current_time=float("inf"))
    assert cmd_inf_t.state in [SupervisorState.SLOWDOWN, SupervisorState.EMERGENCY_STOP]
    assert cmd_inf_t.state != SupervisorState.NORMAL

    # 4. Negative computed age
    tac_future = TacticalTelemetry(p10=0.05, d10_clear=0.30, timestamp=10.0)
    cmd_future = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_future, current_time=1.0)
    assert cmd_future.state in [SupervisorState.SLOWDOWN, SupervisorState.EMERGENCY_STOP]
    assert cmd_future.state != SupervisorState.NORMAL

    # 5. Out of range P10 (> 1.0)
    tac_bad_p = TacticalTelemetry(p10=1.5, d10_clear=0.30, timestamp=0.10)
    cmd_bad_p = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_bad_p, current_time=0.11)
    assert cmd_bad_p.state in [SupervisorState.SLOWDOWN, SupervisorState.EMERGENCY_STOP]
    assert cmd_bad_p.state != SupervisorState.NORMAL


# =============================================================================
# 4. Moderate Risk Predictive Slowdown Escalation
# =============================================================================
def test_predictive_slowdown_escalation():
    """Verify that confirmed moderate risk (2-of-3 ticks) escalates NORMAL to SLOWDOWN."""
    supervisor = SafetySupervisor(config={"require_h50": False})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.35)

    # Moderate risk: P10 = 0.45 >= 0.36
    tac_mod = TacticalTelemetry(p10=0.45, d10_clear=0.15, timestamp=0.10, age_ms=0.0)

    # Tick 1: filter receives 1st hit -> CAUTION
    cmd1 = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_mod, current_time=0.10)
    assert cmd1.state in [SupervisorState.CAUTION, SupervisorState.SLOWDOWN]

    # Tick 2: filter receives 2nd hit (2-of-3 confirmed) -> SLOWDOWN
    cmd2 = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_mod, current_time=0.11)
    assert cmd2.state == SupervisorState.SLOWDOWN
    assert cmd2.stop_mode == StopMode.CONTROLLED_DECEL


# =============================================================================
# 5. Critical Fast-Path Predictive Intervention
# =============================================================================
def test_predictive_critical_fast_path():
    """Verify that critical risk (P10 >= 0.70 or d10 <= 0.03m) triggers 0ms immediate SLOWDOWN."""
    supervisor = SafetySupervisor(config={"require_h50": False})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.35)

    # Critical fast-path: P10 = 0.85 >= 0.70 (0 ms filter delay)
    tac_crit = TacticalTelemetry(p10=0.85, d10_clear=0.05, timestamp=0.10, age_ms=0.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_safe, tactical=tac_crit, current_time=0.10)
    assert cmd.state == SupervisorState.SLOWDOWN
    assert any("CRITICAL FAST-PATH" in a.upper() for a in cmd.active_alarms)


# =============================================================================
# 6. Layer 0 Reactive Override Priority
# =============================================================================
def test_reactive_l0_unconditional_override():
    """Verify that Layer 0 collision unconditionally overrides any predictive state and latches."""
    supervisor = SafetySupervisor(config={"require_h50": False})
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.zeros(7)

    # Even if tactical telemetry claims low risk:
    tac_safe = TacticalTelemetry(p10=0.01, d10_clear=0.50, timestamp=0.10, age_ms=0.0)

    # Layer 0 contact occurs:
    reactive_collision = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)

    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive=reactive_collision, tactical=tac_safe, current_time=0.10)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.stop_mode == StopMode.LATCHED_REACTIVE


# =============================================================================
# 7. Actuator Inhibition Recovery Protection
# =============================================================================
def test_actuator_inhibition_no_bypass(shared_env: SawmEnvironment):
    """Verify that latched actuator inhibition cannot be bypassed autonomously without valid operator handshake."""
    supervisor = SafetySupervisor()
    shared_env.reset()

    # Inhibit actuators under supervisor
    shared_env.inhibit_arm_actuators(supervisor=supervisor)
    assert shared_env.robot.arm_actuators_inhibited is True

    # Attempt recovery with None or invalid supervisor -> REJECTED
    ok, msg = shared_env.authorize_actuator_recovery(None)
    assert ok is False
    assert "REJECTED" in msg
    assert shared_env.robot.arm_actuators_inhibited is True

    # Attempt recovery while supervisor is latched -> REJECTED
    supervisor._is_latched = True
    ok, msg = shared_env.authorize_actuator_recovery(supervisor)
    assert ok is False
    assert "REJECTED" in msg
    assert shared_env.robot.arm_actuators_inhibited is True

    # Clean up
    supervisor._is_latched = False
    supervisor._has_completed_operator_recovery = True
    supervisor._state = SupervisorState.RECOVERY
    shared_env.authorize_actuator_recovery(supervisor)
    assert shared_env.robot.arm_actuators_inhibited is False


# =============================================================================
# 8. H10Predictor Online State Tracking & Forward Rollout Sanity
# =============================================================================
def test_h10_predictor_online_parity_rollout(shared_env: SawmEnvironment):
    """Verify that H10Predictor properly initializes, tracks state, and imagines rollouts."""
    predictor = H10Predictor()
    obs = shared_env.reset()
    t0 = float(obs["time"])

    predictor.reset(shared_env, obs, current_time=t0)
    assert predictor.h_t is not None
    assert predictor.s_t is not None
    assert predictor.h_t.shape == (1, 128)
    assert predictor.s_t.shape == (1, 32)

    # Nominal 10-step action sequence
    actions_10 = np.tile(np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0], dtype=np.float32), (10, 1))
    tac = predictor.predict_h10(actions_10, current_time=t0, raw_obs=obs)

    assert isinstance(tac, TacticalTelemetry)
    assert np.isfinite(tac.p10)
    assert np.isfinite(tac.d10_clear)
    assert 0.0 <= tac.p10 <= 1.0
    assert tac.d10_clear >= 0.0
    assert tac.timestamp == t0

    # Step causal update with dummy action
    act_8d = actions_10[0]
    next_obs = shared_env.step(arm_targets=act_8d[:7])
    t1 = float(next_obs["time"])
    predictor.update(act_8d, next_obs, current_time=t1)
    assert predictor.current_time == t1


# =============================================================================
# 9. Scenario A Demonstration Verification (Nominal Pick-and-Place)
# =============================================================================
def test_demo_scenario_a_completion(shared_env: SawmEnvironment):
    """Verify that Scenario A executes to 100% completion with zero false-positive interventions."""
    res_a = run_scenario_a_normal(shared_env, capture_visuals=False)
    assert res_a["task_success"] is True, f"Scenario A failed: placement error={res_a['placement_error_mm']:.2f}mm"
    assert res_a["placement_error_mm"] < 10.0, "Placement error should be within 10mm"
    assert res_a["non_normal_interventions"] == 0, f"Expected zero interventions, got {res_a['non_normal_interventions']}"
    assert res_a["final_supervisor_state"] == "NORMAL"
    assert res_a["peak_p10"] < 0.36, "P10 should stay well below alarm threshold in nominal run"


# =============================================================================
# 10. Scenario B Demonstration Verification (Predictive Hazard Intervention)
# =============================================================================
def test_demo_scenario_b_intervention(shared_env: SawmEnvironment):
    """Verify that Scenario B triggers predictive slowdown and brings robot to rest before Layer 0 contact."""
    res_b = run_scenario_b_predictive_hazard(shared_env, capture_visuals=False)
    assert res_b["intervention_step"] is not None, "Predictive intervention must be triggered"
    assert res_b["intervention_state"] in ["SLOWDOWN", "CAUTION"], f"Unexpected intervention state: {res_b['intervention_state']}"
    assert res_b["stopped_step"] is not None, "Robot must be brought to rest before collision"
    assert res_b["layer0_collision_occurred"] is False, "Physical collision must be completely avoided"
    assert res_b["geom_dist_at_stop_mm"] > 10.0, f"Clearance at rest ({res_b['geom_dist_at_stop_mm']:.1f}mm) must exceed 10mm"


# =============================================================================
# 11. Horizon Ordering & Ground-Truth Alignment Verification
# =============================================================================
def test_horizon_ordering_and_ground_truth_alignment(shared_env: SawmEnvironment):
    """
    Verify reverse-cumulative monotonicity and ground-truth alignment across horizons:
    1. Output clearances strictly non-increasing across all horizons (H1 >= H5 >= H10 >= H20 >= H50).
    2. Output probabilities strictly non-decreasing across all horizons (H1 <= H5 <= H10 <= H20 <= H50).
    3. H10 >= H50 verified on the exact same action rollout sequence.
    4. Predicted clearance acts as a conservative lower bound relative to ground truth.
    """
    predictor = H10Predictor()
    obs = shared_env.reset()

    # Encroaching proxy
    p_start = np.array([0.80, 0.25, 0.60], dtype=np.float64)
    p_target = np.array([0.52, 0.05, 0.58], dtype=np.float64)
    shared_env.obstacle_manager.set_proxy_trajectory(p_start, p_target, steps=120)
    obs = shared_env.get_observation()

    t0 = float(obs["time"])
    predictor.reset(shared_env, obs, current_time=t0)

    planner = BaselinePlanner(shared_env)
    keyframes, ok, _ = planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
    assert ok
    planned_steps = planner.build_discrete_plan(keyframes)

    # Step to Scenario B Step 2
    for s_idx in range(2):
        act = planned_steps[s_idx]["action_vector"]
        obs = shared_env.step(arm_targets=planned_steps[s_idx]["arm_target"])
        predictor.update(act, obs, current_time=float(obs["time"]))

    # Test A: 50-step action sequence (unclamped horizons [0, 4, 9, 19, 49])
    a50, _ = planner.get_planned_action_sequence(50, current_cursor=2)
    act_t50 = torch.from_numpy(a50).float().unsqueeze(0)
    _, h50, s50 = predictor.rssm.imagine(predictor.h_t, predictor.s_t, act_t50)
    z50 = torch.cat([h50, s50], dim=-1)
    out50 = predictor.physical_head(z50, use_temperature=True)

    c50 = out50["clearances"][0].detach().cpu().numpy()
    p50 = out50["collision_probs"][0].detach().cpu().numpy()

    # Verify non-increasing clearance across all 5 horizons
    assert c50[0] >= c50[1] - 1e-6, f"H1 ({c50[0]}) < H5 ({c50[1]})"
    assert c50[1] >= c50[2] - 1e-6, f"H5 ({c50[1]}) < H10 ({c50[2]})"
    assert c50[2] >= c50[3] - 1e-6, f"H10 ({c50[2]}) < H20 ({c50[3]})"
    assert c50[3] >= c50[4] - 1e-6, f"H20 ({c50[3]}) < H50 ({c50[4]})"
    assert c50[2] >= c50[4], f"H10 ({c50[2]*1000:.1f}mm) must be >= H50 ({c50[4]*1000:.1f}mm) on same rollout!"

    # Verify non-decreasing collision probabilities
    assert p50[0] <= p50[1] + 1e-6
    assert p50[1] <= p50[2] + 1e-6
    assert p50[2] <= p50[3] + 1e-6
    assert p50[3] <= p50[4] + 1e-6

    # Test B: 10-step action sequence
    a10, _ = planner.get_planned_action_sequence(10, current_cursor=2)
    act_t10 = torch.from_numpy(a10).float().unsqueeze(0)
    _, h10, s10 = predictor.rssm.imagine(predictor.h_t, predictor.s_t, act_t10)
    z10 = torch.cat([h10, s10], dim=-1)
    out10 = predictor.physical_head(z10, use_temperature=True)

    c10 = out10["clearances"][0].detach().cpu().numpy()
    assert c10[0] >= c10[1] - 1e-6
    assert c10[1] >= c10[2] - 1e-6
    assert c10[2] >= c10[3] - 1e-6
    assert c10[3] >= c10[4] - 1e-6
    assert c10[2] >= c10[4], f"H10 ({c10[2]*1000:.1f}mm) must be >= H50 ({c10[4]*1000:.1f}mm) on 10-step rollout!"

    # Test C: Ground-truth comparison
    # Step forward 10 steps to measure actual kinematic minimum clearance
    actual_10_clearances = []
    for f_idx in range(2, 12):
        f_obs = shared_env.step(arm_targets=planned_steps[f_idx]["arm_target"])
        actual_10_clearances.append(float(f_obs["distances"].geometric_min_distance))
    actual_min_10 = min(actual_10_clearances)

    # Predicted clearance at H=10 must be a conservative safety bound (pred <= actual)
    assert c50[2] <= actual_min_10, f"Predicted H10 ({c50[2]*1000:.1f}mm) should be conservative relative to actual ({actual_min_10*1000:.1f}mm)"

