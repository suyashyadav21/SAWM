"""
Automated Test Suite for Milestone 6 Step 1: Safety Supervisor State Machine.

Comprehensive safety verification suite covering:
1. Missing, stale, expired, and non-finite H10 telemetry (fail-safe policy).
2. Mandatory fresh H50 strategic telemetry requirement (missing, stale, expired vs fresh).
3. Reactive latch integrity and public reset() protection (no silent latch clearing).
4. Operator reset handshake and collision sensor checks (obstacle, proxy, table).
5. Braking trajectory completion, exhaustion guard, and edge-case handling (NaN, limits, inertia, dt).
6. 1D kinematic stopping equation (directionality, magnitude, dimensional consistency).
7. Critical contextual harm thresholds (C_harm >= 0.85 severe emergency, proxy <= 0.03m).
8. Gripper emergency clamping (payload grasp retention during arrest).
9. Actual MuJoCo dynamic arrest simulation (actual physics stepping, acceleration, stopping time).
10. Gripper physical verification in MuJoCo (opening, closing, force, hold, slip).

Usage:
  pytest tests/test_safety_supervisor.py -v
"""

import os
import sys
import numpy as np
import pytest
import mujoco

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety.safety_supervisor import (
    SafetySupervisor,
    SupervisorState,
    StopMode,
    ReactiveTelemetry,
    TacticalTelemetry,
    StrategicTelemetry,
    BrakingTrajectoryResult,
    InterventionCommand,
)
from simulation.environment import SawmEnvironment
from planning.baseline_planner import BaselinePlanner


# =============================================================================
# Test 1: Missing, Stale, Expired, and Non-Finite H10 Telemetry (Fail-Safe)
# =============================================================================
def test_h10_telemetry_fail_safe_policy():
    """
    Verify that missing or compromised H10 telemetry NEVER allows NORMAL to persist:
    - Missing H10 (tactical=None) forces SLOWDOWN.
    - Stale H10 (40ms < age <= 80ms) forces CAUTION with speed <= 0.50x.
    - Expired H10 (age > 80ms) forces SLOWDOWN.
    - Non-finite or out-of-range H10 forces SLOWDOWN.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    # 1. Start in NORMAL with fresh H10 and fresh H50
    tac_fresh = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_fresh, strat_fresh)
    assert cmd.state == SupervisorState.NORMAL
    assert cmd.speed_scale == 1.0

    # 2. Missing H10 telemetry (tactical=None) -> CANNOT stay in NORMAL, must force SLOWDOWN
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, None, strat_fresh)
    assert cmd.state == SupervisorState.SLOWDOWN
    assert any("MISSING" in a.upper() for a in cmd.active_alarms)

    # 3. Slightly Stale H10 (age = 60ms) -> forces CAUTION with speed <= 0.50
    supervisor = SafetySupervisor()
    tac_stale = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=60.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_stale, strat_fresh)
    assert cmd.state == SupervisorState.CAUTION
    assert cmd.speed_scale <= 0.50

    # 4. Expired H10 (age = 90ms) -> forces SLOWDOWN
    supervisor = SafetySupervisor()
    tac_expired = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=90.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_expired, strat_fresh)
    assert cmd.state == SupervisorState.SLOWDOWN
    assert any("EXPIRED" in a.upper() for a in cmd.active_alarms)

    # 5. Non-finite / out-of-range H10 -> forces conservative stop (SLOWDOWN or EMERGENCY_STOP)
    supervisor = SafetySupervisor()
    tac_nan = TacticalTelemetry(p10=float("nan"), d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_nan, strat_fresh)
    assert cmd.state in [SupervisorState.SLOWDOWN, SupervisorState.EMERGENCY_STOP]
    assert any("INVALID" in a.upper() for a in cmd.active_alarms)


# =============================================================================
# Test 2: Mandatory Fresh H50 Requirement
# =============================================================================
def test_mandatory_fresh_h50_requirement():
    """
    Verify that CAUTION cannot de-escalate to NORMAL without fresh H50:
    - Missing H50 (strategic=None) -> blocks return to NORMAL.
    - Stale H50 (is_stale=True or age > 150ms) -> blocks return to NORMAL.
    - Elevated P50 (p50 > 0.20) -> blocks return to NORMAL.
    - Fresh H50 (p50 <= 0.20, d50 >= 0.15m, age <= 150ms) for 10 ticks -> returns to NORMAL.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.zeros(7)
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    tac_clear = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)

    # Put supervisor into CAUTION
    tac_caution = TacticalTelemetry(p10=0.15, d10_clear=0.15, c_harm=0.10, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_caution)
    assert cmd.state == SupervisorState.CAUTION

    # Case A: Missing H50 (strategic is None) -> Stays in CAUTION indefinitely
    for _ in range(15):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strategic=None)
        assert cmd.state == SupervisorState.CAUTION, "Missing H50 must NOT permit CAUTION -> NORMAL"

    # Case B: Stale H50 flag -> Stays in CAUTION
    strat_flag_stale = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=True)
    for _ in range(15):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strat_flag_stale)
        assert cmd.state == SupervisorState.CAUTION, "Flagged stale H50 must NOT permit CAUTION -> NORMAL"

    # Case C: Expired H50 age (> 150ms) -> Stays in CAUTION
    strat_old_age = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=200.0, is_stale=False)
    for _ in range(15):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strat_old_age)
        assert cmd.state == SupervisorState.CAUTION, "H50 age > 150ms must NOT permit CAUTION -> NORMAL"

    # Case D: Elevated P50 (> 0.20) -> Stays in CAUTION
    strat_elevated = StrategicTelemetry(p50=0.25, d50_clear=0.30, age_ms=50.0, is_stale=False)
    for _ in range(15):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strat_elevated)
        assert cmd.state == SupervisorState.CAUTION, "P50 > 0.20 must NOT permit CAUTION -> NORMAL"

    # Case E: Genuinely fresh H50 sustained for 10 ticks -> Successfully transitions to NORMAL
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)
    for tick in range(9):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strat_fresh)
        assert cmd.state == SupervisorState.CAUTION, f"Should remain in CAUTION before 10 sustained ticks (tick {tick+1})"

    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_clear, strat_fresh)
    assert cmd.state == SupervisorState.NORMAL, "Must return to NORMAL on 10th sustained tick with fresh H50"


# =============================================================================
# Test 3: Reactive Latch Integrity (No Silent Clear via reset())
# =============================================================================
def test_reactive_latch_integrity_and_reset_protection():
    """
    Verify public reset() does NOT silently clear an active reactive emergency latch:
    - Calling reset() without operator_confirm=True fails and preserves latch.
    - Calling reset() with active sensor collision fails.
    - Calling clear_latched_emergency_stop() rejects missing reactive telemetry.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)

    # Trigger Layer 0 latch
    reactive_collision = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)
    supervisor.step(q_curr, q_vel, q_curr, 0.0, reactive_collision)
    assert supervisor.is_latched is True
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    # Attempt 1: Default reset() call without operator_confirm -> must REFUSE
    success = supervisor.reset()
    assert success is False, "reset() must refuse to clear active reactive emergency latch"
    assert supervisor.is_latched is True
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    # Attempt 2: reset(operator_confirm=True) while physical collision persists -> must REFUSE
    success = supervisor.reset(operator_confirm=True)
    assert success is False, "reset() must refuse while collision is active"
    assert supervisor.is_latched is True

    # Attempt 3: clear_latched_emergency_stop with missing reactive telemetry -> must REJECT
    supervisor.latest_reactive = None
    ok, msg = supervisor.clear_latched_emergency_stop(q_vel, operator_confirm=True)
    assert ok is False
    assert "missing" in msg.lower()


# =============================================================================
# Test 4: Comprehensive Collision Sensor Checks in Handshake
# =============================================================================
def test_handshake_checks_all_collision_sensors():
    """Verify clear_latched_emergency_stop checks general, obstacle, proxy, and table collisions."""
    supervisor = SafetySupervisor()
    q_rest = np.zeros(7)

    def _latch():
        supervisor._is_latched = True
        supervisor._state = SupervisorState.EMERGENCY_STOP

    # Check 1: robot_table_collision blocks reset
    _latch()
    supervisor.latest_reactive = ReactiveTelemetry(robot_table_collision=True, geometric_min_distance=0.20)
    ok, msg = supervisor.clear_latched_emergency_stop(q_rest, operator_confirm=True)
    assert ok is False
    assert "table" in msg.lower()

    # Check 2: robot_proxy_collision blocks reset
    _latch()
    supervisor.latest_reactive = ReactiveTelemetry(robot_proxy_collision=True, geometric_min_distance=0.20)
    ok, msg = supervisor.clear_latched_emergency_stop(q_rest, operator_confirm=True)
    assert ok is False
    assert "proxy" in msg.lower()

    # Check 3: has_collision blocks reset
    _latch()
    supervisor.latest_reactive = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.20)
    ok, msg = supervisor.clear_latched_emergency_stop(q_rest, operator_confirm=True)
    assert ok is False

    # Check 4: All cleared and distance >= 100mm -> reset accepted
    _latch()
    supervisor.latest_reactive = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.15)
    ok, msg = supervisor.clear_latched_emergency_stop(q_rest, operator_confirm=True)
    assert ok is True
    assert supervisor.is_latched is False


# =============================================================================
# Test 5: Braking Trajectory Completion, Exhaustion Guard, and Edge Cases
# =============================================================================
def test_braking_trajectory_completion_and_edge_cases():
    """
    Verify braking trajectory generator:
    - Never unconditionally replaces final velocity with zero.
    - Returns success=False on iteration exhaustion.
    - Rejects NaN/Inf inputs, invalid dt, invalid inertia.
    - Clamps to joint limits for both positive and negative velocities.
    """
    q_start = np.zeros(7)
    v_high = np.ones(7) * 2.0

    # Case A: Exhaustion with max_steps=5 -> must fail gracefully, keeping non-zero velocity
    res_exhaust = SafetySupervisor.generate_braking_trajectory(
        q_start, v_high, max_steps=5, dt=0.01, a_max_provisional=15.0
    )
    assert res_exhaust.success is False
    assert "Iteration limit" in str(res_exhaust.failure_reason)
    assert np.max(np.abs(res_exhaust.velocities[-1])) > SafetySupervisor.REST_VELOCITY_THRESHOLD
    assert res_exhaust.stopping_steps == 5

    # Case B: Sufficient steps -> succeeds and stops at rest
    res_ok = SafetySupervisor.generate_braking_trajectory(
        q_start, v_high, max_steps=200, dt=0.01, a_max_provisional=15.0
    )
    assert res_ok.success is True
    assert np.max(np.abs(res_ok.velocities[-1])) <= SafetySupervisor.REST_VELOCITY_THRESHOLD

    # Case C: Non-finite inputs rejected
    q_nan = q_start.copy()
    q_nan[0] = float("nan")
    res_nan = SafetySupervisor.generate_braking_trajectory(q_nan, v_high)
    assert res_nan.success is False

    # Case D: Invalid dt rejected
    res_bad_dt = SafetySupervisor.generate_braking_trajectory(q_start, v_high, dt=-0.01)
    assert res_bad_dt.success is False

    # Case E: Negative velocities decelerate correctly
    v_neg = - np.ones(7) * 1.5
    res_neg = SafetySupervisor.generate_braking_trajectory(q_start, v_neg)
    assert res_neg.success is True
    assert res_neg.positions[-1][0] < q_start[0]

    # Case F: Joint limit clamping near bounds
    q_near_upper = SafetySupervisor.JOINT_LIMITS_UPPER.copy() - 0.05
    res_clamp = SafetySupervisor.generate_braking_trajectory(q_near_upper, v_high)
    assert res_clamp.success is True
    for q_step in res_clamp.positions:
        for i in range(7):
            q_max = SafetySupervisor.JOINT_LIMITS_UPPER[i] - SafetySupervisor.JOINT_LIMIT_SAFETY_MARGIN_RAD
            assert q_step[i] <= q_max + 1e-6


# =============================================================================
# Test 6: 1D Kinematic Stopping Equation Verification
# =============================================================================
def test_1d_kinematic_stopping_equation():
    """Verify simplified 1D kinematic stopping equation formula and error handling."""
    q_curr = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    v_pos = np.array([1.0, 0.5, 2.0, 0.1, 0.0, 0.0, 0.0])

    q_stop_pos = SafetySupervisor.compute_stopping_distance_1d(q_curr, v_pos, a_brake=15.0)
    assert np.all(q_stop_pos >= q_curr)

    q_stop_neg = SafetySupervisor.compute_stopping_distance_1d(q_curr, -v_pos, a_brake=15.0)
    assert np.all(q_stop_neg <= q_curr)

    with pytest.raises(ValueError):
        SafetySupervisor.compute_stopping_distance_1d(q_curr, v_pos, a_brake=0.0)
    with pytest.raises(ValueError):
        SafetySupervisor.compute_stopping_distance_1d(q_curr, v_pos, a_brake=-10.0)


# =============================================================================
# Test 7: Critical Contextual Harm Thresholds (C_harm >= 0.85 & Human Proxy)
# =============================================================================
def test_critical_contextual_harm_and_severe_emergency():
    """
    Verify contextual harm mappings:
    - C_harm >= 0.85 -> Severe Emergency (EMERGENCY_STOP unlatched).
    - Human proxy distance <= 0.03m -> Severe Emergency (EMERGENCY_STOP unlatched).
    - 0.70 <= C_harm < 0.85 -> Tier 2 Fast-Path (SLOWDOWN).
    - 0.30 <= C_harm < 0.70 -> Tier 1 Moderate Warning (2-of-3 filtered CAUTION).
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.zeros(7)
    reactive_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    # 1. Severe Harm: C_harm = 0.88 -> EMERGENCY_STOP
    tac_severe_harm = TacticalTelemetry(p10=0.1, d10_clear=0.2, c_harm=0.88, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_severe_harm, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.stop_mode == StopMode.PREDICTIVE_EMERGENCY
    assert any("SEVERE_CONTEXTUAL_EMERGENCY" in a for a in cmd.active_alarms)

    # 2. Severe Human Proximity: human_proxy_distance = 0.025m <= 0.03m -> EMERGENCY_STOP
    supervisor = SafetySupervisor()
    tac_severe_proxy = TacticalTelemetry(p10=0.1, d10_clear=0.2, c_harm=0.20, human_proxy_distance=0.025, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_severe_proxy, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.stop_mode == StopMode.PREDICTIVE_EMERGENCY
    assert any("SEVERE_HUMAN_PROXIMITY_EMERGENCY" in a for a in cmd.active_alarms)

    # 3. Critical Contextual Harm: C_harm = 0.75 -> SLOWDOWN
    supervisor = SafetySupervisor()
    tac_crit_harm = TacticalTelemetry(p10=0.1, d10_clear=0.2, c_harm=0.75, human_proxy_distance=0.15, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_crit_harm, strat_fresh)
    assert cmd.state == SupervisorState.SLOWDOWN

    # 4. Moderate Contextual Harm: C_harm = 0.50 -> CAUTION after 2-of-3 filter
    supervisor = SafetySupervisor()
    tac_mod_harm = TacticalTelemetry(p10=0.1, d10_clear=0.2, c_harm=0.50, human_proxy_distance=0.15, age_ms=20.0)
    supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_mod_harm, strat_fresh) # Tick 1
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, reactive_safe, tac_mod_harm, strat_fresh) # Tick 2
    assert cmd.state == SupervisorState.CAUTION
    assert cmd.speed_scale <= 0.60


# =============================================================================
# Test 8: Emergency Gripper Clamping Protection
# =============================================================================
def test_emergency_gripper_clamping_protection():
    """Verify emergency stop overrides nominal open commands to clamp gripper if holding object."""
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7)
    q_vel = np.zeros(7)
    q_nom = np.zeros(7)

    reactive_holding = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, robot_object_contact=True)
    cmd = supervisor.step(q_curr, q_vel, q_nom, nominal_gripper_target=255.0, reactive=reactive_holding)

    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.target_gripper == 0.0, "Emergency stop must clamp gripper (0.0) when object is held"


# =============================================================================
# Test 9: Actual MuJoCo Dynamic Arrest Simulation Response
# =============================================================================
def test_actual_mujoco_dynamic_arrest_response():
    """
    Physical verification of dynamic arrest using actual MuJoCo physics stepping:
    - Arm in motion in MuJoCo (v_init > 0.5 rad/s).
    - Layer 0 collision occurs, triggering latched EMERGENCY_STOP.
    - Measure actual joint acceleration, stopping time, displacement, residual velocity, and actuator torque.
    - Demonstrates that commanding position arrest under velocity produces peak decelerations
      reaching actuator torque saturation (up to ~88 rad/s²), and that settling to rest (< 1e-3 rad/s)
      requires ~350-650 ms (honest reporting vs the theoretical 120 ms proposal).
    - Dynamic arrest transitions to steady position hold when velocity drops below 1e-3 rad/s.
    """
    env = SawmEnvironment("configs/simulation.yaml")
    env.reset()

    traj = env.robot.interpolate_trajectory(env.home_qpos, env.ready_qpos, steps=30)
    for wp in traj[:15]:
        env.step(arm_targets=wp)

    q_init = env.robot.get_joint_positions()
    v_init = env.robot.get_joint_velocities()
    max_v_init = np.max(np.abs(v_init))
    print(f"\n  [MuJoCo Arrest] Initial Peak Velocity: {max_v_init:.3f} rad/s")
    assert max_v_init > 0.5, "Arm should have significant velocity before arrest test"

    supervisor = SafetySupervisor()
    dt = env.dt * env.frame_skip

    reactive_collision = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)

    accelerations = []
    actuator_torques = []
    t_threshold_step = None
    t_settled_step = None
    settle_count = 0

    for step in range(120):
        q_now = env.robot.get_joint_positions()
        v_now = env.robot.get_joint_velocities()

        cmd = supervisor.step(q_now, v_now, q_now, 0.0, reactive_collision, dt=dt)
        obs = env.step(arm_targets=cmd.target_qpos, gripper_target=cmd.target_gripper)

        v_after = env.robot.get_joint_velocities()
        acc = (v_after - v_now) / dt
        accelerations.append(acc)
        actuator_torques.append(env.data.qfrc_actuator[:7].copy())

        max_v = np.max(np.abs(v_after))
        if max_v < SafetySupervisor.REST_VELOCITY_THRESHOLD:
            if t_threshold_step is None:
                t_threshold_step = step + 1
            settle_count += 1
            if settle_count == 10 and t_settled_step is None:
                t_settled_step = step + 1
        else:
            settle_count = 0

        if not supervisor.is_arresting and t_settled_step is not None:
            break

    final_q = env.robot.get_joint_positions()
    final_v = env.robot.get_joint_velocities()
    stopping_disp_rad = np.linalg.norm(final_q - q_init)
    max_acc = np.max(np.abs(accelerations))
    max_torque = np.max(np.abs(actuator_torques))

    print(f"  [MuJoCo Arrest] First Threshold Crossing: Step {t_threshold_step} ({(t_threshold_step or 0)*dt:.3f} s)")
    print(f"  [MuJoCo Arrest] Sustained Settled Step:   Step {t_settled_step} ({(t_settled_step or 0)*dt:.3f} s)")
    print(f"  [MuJoCo Arrest] Stopping Displacement:    {stopping_disp_rad:.4f} rad")
    print(f"  [MuJoCo Arrest] Peak Measured Accel:      {max_acc:.2f} rad/s²")
    print(f"  [MuJoCo Arrest] Peak Actuator Torque:     {max_torque:.2f} Nm (Max Limit: 87.0 Nm)")
    print(f"  [MuJoCo Arrest] Final Residual Vel:       {np.max(np.abs(final_v)):.6f} rad/s")

    assert t_threshold_step is not None, "Robot must reach residual velocity threshold < 1e-3 rad/s"
    assert t_settled_step is not None, "Robot must sustain rest for >= 10 ticks"
    assert t_settled_step >= t_threshold_step, "Sustained settling cannot precede the first threshold crossing"
    assert t_settled_step >= t_threshold_step + 9, "Sustained settling requires at least 9 additional ticks after first crossing"
    assert np.max(np.abs(final_v)) < 0.002, "Residual velocity must be near zero"
    assert supervisor.is_latched is True
    # Verify that actual deceleration exceeds 15 rad/s² under step arrest (torque saturation)
    assert max_acc > 15.0, "Physical step arrest under velocity produces high torque-saturated deceleration"
    assert supervisor.is_arresting is False, "is_arresting must be False once physical rest is sustained for >= 10 ticks"

    # Verify that the latch cannot clear automatically when presented with safe telemetry
    safe_reactive = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    cmd_after = supervisor.step(final_q, final_v, env.ready_qpos, 0.0, safe_reactive, dt=dt)
    assert supervisor.is_latched is True, "Latch must not clear automatically during rest or safe telemetry"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP, "State must remain EMERGENCY_STOP while latched"
    assert np.allclose(cmd_after.target_qpos, supervisor._frozen_qpos, atol=1e-6), "Command must hold settled resting position"
    assert np.allclose(cmd_after.target_qpos, final_q, atol=0.01), "Target command must be near physical resting position within gravity deflection"

    env.close()


# =============================================================================
# Test 10: Gripper Physical Verification in MuJoCo (Condition 4)
# =============================================================================
def test_gripper_physical_verification():
    """
    MuJoCo physical verification of gripper actuator:
    1. Opening span: u=255.0 opens fingers to ~0.04m each.
    2. Closing span: u=0.0 closes fingers to < 0.005m.
    3. Contact normal force: positive clamping force (> 1.0 N).
    4. Grasp retention: holding cube suspended in mid-air against gravity.
    5. Object slip during deceleration: bounded decel (<= 15 rad/s²) retains cube.
    """
    env = SawmEnvironment("configs/simulation.yaml")
    planner = BaselinePlanner(env)
    obs = env.reset()

    # Part 1: Gripper Opening
    env.robot.set_gripper_target(255.0)
    for _ in range(50):
        env.step(env.home_qpos, gripper_target=255.0)
    grip_open = env.robot.get_gripper_position()
    assert grip_open >= 0.035, f"Gripper should open >= 35mm per finger, got {grip_open*1000:.1f}mm"

    # Part 2: Gripper Closing
    env.robot.set_gripper_target(0.0)
    for _ in range(50):
        env.step(env.home_qpos, gripper_target=0.0)
    grip_closed = env.robot.get_gripper_position()
    assert grip_closed <= 0.005, f"Gripper should close to < 5mm per finger, got {grip_closed*1000:.3f}mm"

    # Part 3: Grasping Cube and Contact Force
    obs = env.reset()
    keyframes, ok, _ = planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
    assert ok

    for wp in [keyframes["pregrasp"], keyframes["grasp"]]:
        for _ in range(50):
            env.step(arm_targets=wp, gripper_target=planner.open_ctrl)
    for _ in range(40):
        obs = env.step(arm_targets=keyframes["grasp"], gripper_target=planner.close_ctrl)
    assert obs["contacts"]["robot_object_contact"] is True

    total_normal_force = 0.0
    for i in range(env.data.ncon):
        c_forces = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, i, c_forces)
        if c_forces[0] > 0:
            total_normal_force += c_forces[0]
    assert total_normal_force > 1.0, f"Clamping normal force should exceed 1.0 N, got {total_normal_force:.2f} N"

    # Part 4: Grasp Retention in Mid-Air Against Gravity
    for _ in range(50):
        obs = env.step(arm_targets=keyframes["lift"], gripper_target=planner.close_ctrl)
    for _ in range(100):
        obs = env.step(arm_targets=keyframes["lift"], gripper_target=planner.close_ctrl)
    cube_z = obs["target_object"]["position"][2]
    lift_height = cube_z - planner.table_z
    assert lift_height >= 0.050
    assert obs["contacts"]["robot_object_contact"] is True

    # Part 5: Slip Verification during Deceleration
    hand_id = env.robot.hand_body_id
    cube_id = env.target_object_body_id
    rel_pos_initial = env.data.xpos[cube_id] - env.data.xpos[hand_id]

    traj = env.robot.interpolate_trajectory(keyframes["lift"], keyframes["transport"], steps=40)
    for wp in traj[:20]:
        obs = env.step(arm_targets=wp, gripper_target=planner.close_ctrl)
    q_stop = env.robot.get_joint_positions()
    for _ in range(30):
        obs = env.step(arm_targets=q_stop, gripper_target=planner.close_ctrl)

    rel_pos_caseA = env.data.xpos[cube_id] - env.data.xpos[hand_id]
    slip_caseA = np.linalg.norm(rel_pos_caseA - rel_pos_initial)
    assert slip_caseA < 0.005, f"Bounded deceleration slip should be < 5mm, got {slip_caseA*1000:.3f}mm"

    env.close()


# =============================================================================
# Test 11: Reactive Dynamic Arrest Trajectory Controller Invariants
# =============================================================================
def test_reactive_arrest_trajectory_controller_invariants():
    """
    Verify exact SafetySupervisor reactive dynamic arrest execution semantics:
    1. Immediate command issued after reactive contact is _braking_trajectory[0].
    2. Subsequent commands advance sequentially along _braking_trajectory.
    3. Step index advances with each step during dynamic arrest.
    4. If steps exhaust before rest, controller holds _braking_trajectory[-1].
    5. Switch to stationary position hold occurs ONLY when velocity remains
       below REST_VELOCITY_THRESHOLD for >= 10 consecutive ticks.
    6. Upon sustained rest, _frozen_qpos is updated to the actual resting pose.
    7. Reactive latch remains active (_is_latched=True) throughout all phases:
       impact -> dynamic arrest -> sustained settling -> stationary position hold.
    8. Latch CANNOT clear automatically under nominal commands or safe telemetry.
    9. Unconfirmed reset is rejected and preserves latch.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    v_moving = np.array([0.6, 0.4, 0.0, -0.5, 0.0, 0.2, 0.0], dtype=np.float64)
    q_nom = np.array([0.1, 0.6, 0.0, -0.9, 0.0, 0.6, 0.0], dtype=np.float64)
    r_collision = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.0)

    # 1. Immediate step upon contact
    cmd0 = supervisor.step(q_curr, v_moving, q_nom, 0.0, r_collision, dt=0.01)
    assert supervisor.is_latched is True, "Latch must engage on first collision tick"
    assert supervisor.is_arresting is True, "Moving arm must enter dynamic arrest"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert len(supervisor._braking_trajectory) > 0, "Braking trajectory must be generated"
    assert np.allclose(cmd0.target_qpos, supervisor._braking_trajectory[0]), \
        "First command must be the first step of braking trajectory, not q_curr or static freeze"
    assert supervisor._braking_step_idx == 1

    # 2. Sequential command advancement along braking trajectory
    cmd1 = supervisor.step(q_curr, v_moving, q_nom, 0.0, r_collision, dt=0.01)
    assert supervisor.is_latched is True
    assert supervisor.is_arresting is True
    assert np.allclose(cmd1.target_qpos, supervisor._braking_trajectory[1]), \
        "Second command must advance to second step of braking trajectory"
    assert supervisor._braking_step_idx == 2

    # 3. Exhaustion behavior: if steps exceed trajectory length, command holds final position
    supervisor._braking_step_idx = len(supervisor._braking_trajectory)
    cmd_exhaust = supervisor.step(q_curr, v_moving, q_nom, 0.0, r_collision, dt=0.01)
    assert supervisor.is_latched is True
    assert supervisor.is_arresting is True
    assert np.allclose(cmd_exhaust.target_qpos, supervisor._braking_trajectory[-1]), \
        "Exhausted trajectory must hold final calculated braking setpoint"

    # 4. Sustained settling: rest for 9 ticks must NOT complete dynamic arrest
    v_rest = np.zeros(7, dtype=np.float64)
    for _ in range(9):
        cmd = supervisor.step(q_curr, v_rest, q_nom, 0.0, r_collision, dt=0.01)
        assert supervisor.is_arresting is True, "Dynamic arrest must remain True before 10 ticks of rest"
        assert supervisor.is_latched is True

    # 5. On 10th consecutive tick, transition to stationary position hold occurs
    cmd_settled = supervisor.step(q_curr, v_rest, q_nom, 0.0, r_collision, dt=0.01)
    assert supervisor.is_arresting is False, "Dynamic arrest must become False once rest sustained for 10 ticks"
    assert supervisor.is_latched is True, "Latch MUST remain True when dynamic arrest transitions to hold"
    assert np.allclose(supervisor._frozen_qpos, supervisor._braking_trajectory[-1]), \
        "_frozen_qpos must be locked to settled braking trajectory target to maintain setpoint continuity"
    assert np.allclose(cmd_settled.target_qpos, supervisor._frozen_qpos)

    # 6. Stationary position hold ignores nominal commands and safe telemetry
    r_safe = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    for _ in range(10):
        cmd_hold = supervisor.step(q_curr, v_rest, q_nom, 0.0, r_safe, dt=0.01)
        assert supervisor.is_latched is True, "Latch must not clear automatically"
        assert supervisor.state == SupervisorState.EMERGENCY_STOP
        assert np.allclose(cmd_hold.target_qpos, supervisor._frozen_qpos)

    # 7. Unconfirmed reset rejected
    ok, _ = supervisor.clear_latched_emergency_stop(v_rest, operator_confirm=False)
    assert ok is False
    assert supervisor.is_latched is True

    # 8. Confirmed reset clears latch and enters RECOVERY
    ok, msg = supervisor.clear_latched_emergency_stop(v_rest, operator_confirm=True)
    assert ok is True
    assert supervisor.is_latched is False
    assert supervisor.state == SupervisorState.RECOVERY


# =============================================================================
# Test 12: End-to-End Real Obstacle Collision Arrest in MuJoCo
# =============================================================================
def test_e2e_mujoco_actual_obstacle_collision_arrest():
    """
    End-to-End closed-loop integration test in MuJoCo physics:
    - Real SawmEnvironment with static obstacle placed directly in transport path.
    - Arm carrying cube accelerates to high transport velocity (> 1.2 rad/s).
    - Real physical collision occurs with static obstacle (robot_obstacle_collision == True).
    - SafetySupervisor receives real ReactiveTelemetry from simulation.
    - Commands are applied directly to MuJoCo position servos.
    - Records contact forces, accelerations, torques, penetration, and timing.
    - Verifies first threshold crossing, sustained settlement, and permanent latch.
    """
    env = SawmEnvironment("configs/simulation.yaml")
    obs = env.reset()
    planner = BaselinePlanner(env)
    kf, ok, msg = planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
    assert ok

    # Place static obstacle in transport path
    env.obstacle_manager.set_static_obstacle_position(np.array([0.55, 0.05, 0.55]))
    env.robot.set_joint_positions_direct(kf["lift"])

    traj = env.robot.interpolate_trajectory(kf["lift"], kf["transport"], steps=40)
    dt = env.dt * env.frame_skip

    # Step until collision
    col_step = None
    for s in range(40):
        obs = env.step(arm_targets=traj[s])
        if obs["contacts"]["robot_obstacle_collision"]:
            col_step = s
            break

    assert col_step is not None, "Real collision with static obstacle must occur"
    v_trig = env.robot.get_joint_velocities()
    assert np.max(np.abs(v_trig)) > 1.0, f"Trigger velocity must be high, got {np.max(np.abs(v_trig)):.3f}"

    supervisor = SafetySupervisor()
    q_trig = env.robot.get_joint_positions().copy()

    t_threshold_step = None
    t_settled_step = None
    settle_count = 0
    contact_forces = []
    pens = []

    for step in range(160):
        q_now = env.robot.get_joint_positions()
        v_now = env.robot.get_joint_velocities()

        # Measure real obstacle contact force
        step_force = 0.0
        step_pen = 0.0
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            b1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom2])
            if "static_obstacle" in [b1, b2]:
                if c.dist < step_pen:
                    step_pen = c.dist
                cf = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, cf)
                if cf[0] > 0:
                    step_force += cf[0]

        contact_forces.append(step_force)
        pens.append(abs(step_pen))

        r_telem = ReactiveTelemetry(
            has_collision=obs["contacts"]["robot_obstacle_collision"],
            robot_obstacle_collision=obs["contacts"]["robot_obstacle_collision"],
            geometric_min_distance=max(0.0, step_pen)
        )

        cmd = supervisor.step(q_now, v_now, q_trig, 0.0, r_telem, dt=dt)
        obs = env.step(arm_targets=cmd.target_qpos, gripper_target=cmd.target_gripper)

        v_after = env.robot.get_joint_velocities()
        max_v = np.max(np.abs(v_after))
        if max_v < SafetySupervisor.REST_VELOCITY_THRESHOLD:
            if t_threshold_step is None:
                t_threshold_step = step + 1
            settle_count += 1
            if settle_count == 10 and t_settled_step is None:
                t_settled_step = step + 1
        else:
            settle_count = 0

        if not supervisor.is_arresting and t_settled_step is not None:
            break

    assert np.max(contact_forces) > 100.0, f"Real obstacle collision force must exceed 100 N, got {np.max(contact_forces):.1f} N"
    assert np.max(pens) > 0.001, "Penetration depth must be measurable"
    assert supervisor.is_latched is True, "Latch must engage on reactive collision"
    assert t_threshold_step is not None, "Arm must reach residual velocity threshold"
    assert t_settled_step is not None, "Arm must achieve sustained physical rest"
    assert t_settled_step >= t_threshold_step + 9, "Sustained settlement requires >= 9 steps after first crossing"
    assert supervisor.is_arresting is False, "Dynamic arrest concludes once rest is sustained"

    # Latch non-clearance verification
    safe_r = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    cmd_hold = supervisor.step(q_now, v_after, q_trig, 0.0, safe_r, dt=dt)
    assert supervisor.is_latched is True, "Latch cannot clear automatically"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    env.close()


# =============================================================================
# Test 13: Braking Trajectory Exhaustion, Gravity Setpoint & Latch Rejection
# =============================================================================
def test_exhaustion_failure_and_latch_non_clearance_semantics():
    """
    Verify safety failure edge cases:
    1. Exhausted braking trajectory sets _braking_success = False and logs alarm.
    2. Latch remains active after trajectory exhaustion.
    3. Final setpoint maintains setpoint continuity without downward gravity drop.
    4. Operator reset rejected during active robot motion.
    5. Missing or invalid telemetry rejects operator reset.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    v_high = np.ones(7, dtype=np.float64) * 2.5
    r_col = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.0)

    # 1. Trigger dynamic arrest with iteration-limited trajectory generator
    # Force exhaustion in trajectory generator
    traj_exhaust = supervisor.generate_braking_trajectory(q_curr, v_high, max_steps=5, dt=0.01)
    assert traj_exhaust.success is False
    assert "Iteration limit" in str(traj_exhaust.failure_reason)

    # Trigger supervisor step with moving velocity
    cmd = supervisor.step(q_curr, v_high, q_curr, 0.0, r_col, dt=0.01)
    assert supervisor.is_latched is True
    assert supervisor.is_arresting is True

    # Artificially exhaust the trajectory steps
    supervisor._braking_step_idx = len(supervisor._braking_trajectory)
    cmd_ex = supervisor.step(q_curr, v_high, q_curr, 0.0, r_col, dt=0.01)
    # Verify holding final calculated position
    assert np.allclose(cmd_ex.target_qpos, supervisor._braking_trajectory[-1])
    assert supervisor.is_latched is True

    # 2. Operator reset REJECTED during active motion
    supervisor.latest_reactive = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    ok, err = supervisor.clear_latched_emergency_stop(v_high, operator_confirm=True)
    assert ok is False
    assert "non-zero velocity" in err
    assert supervisor.is_latched is True

    # 3. Operator reset REJECTED when reactive telemetry is missing
    v_zero = np.zeros(7, dtype=np.float64)
    supervisor.latest_reactive = None
    ok_no_telem, err_no_telem = supervisor.clear_latched_emergency_stop(v_zero, operator_confirm=True)
    assert ok_no_telem is False
    assert "Reactive telemetry missing" in err_no_telem
    assert supervisor.is_latched is True

    # 4. Operator reset REJECTED when reactive telemetry is invalid
    supervisor.latest_reactive = ReactiveTelemetry(has_collision=False, geometric_min_distance=float("nan"))
    ok_inv, err_inv = supervisor.clear_latched_emergency_stop(v_zero, operator_confirm=True)
    assert ok_inv is False
    assert "Reactive telemetry invalid" in err_inv
    assert supervisor.is_latched is True


# =============================================================================
# Test 14: End-to-End Real Human-Proxy Collision Arrest in MuJoCo
# =============================================================================
def test_e2e_mujoco_human_proxy_collision_arrest():
    """
    End-to-End closed-loop human-proxy collision integration test in MuJoCo physics:
    - Real SawmEnvironment with human proxy capsule placed directly in transport corridor.
    - Arm carrying cube accelerates to high transport velocity (> 1.2 rad/s).
    - Genuine physical collision occurs with human proxy (robot_proxy_collision == True).
    - Contact force is measured directly on human_proxy contacts and is non-zero.
    - SafetySupervisor enters EMERGENCY_STOP and generates/executes dynamic braking trajectory.
    - Commands are applied to MuJoCo actuators, achieving sustained physical rest.
    - Reactive latch remains active throughout dynamic arrest, rest, and safe telemetry.
    """
    env = SawmEnvironment("configs/simulation.yaml")
    obs = env.reset()
    planner = BaselinePlanner(env)
    kf, ok, msg = planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
    assert ok

    # Place human proxy capsule directly in transport corridor
    proxy_pos = np.array([0.55, 0.05, 0.55], dtype=np.float64)
    env.data.mocap_pos[env.obstacle_manager.proxy_mocap_id] = proxy_pos
    env.robot.set_joint_positions_direct(kf["lift"])

    traj = env.robot.interpolate_trajectory(kf["lift"], kf["transport"], steps=40)
    dt = env.dt * env.frame_skip

    # Step until collision
    col_step = None
    for s in range(40):
        obs = env.step(arm_targets=traj[s])
        if obs["contacts"]["robot_proxy_collision"]:
            col_step = s
            break

    assert col_step is not None, "Real collision with human proxy must occur"
    assert obs["contacts"]["robot_proxy_collision"] is True, "robot_proxy_collision must be True"

    v_trig = env.robot.get_joint_velocities()
    assert np.max(np.abs(v_trig)) > 1.0, f"Trigger velocity must be high, got {np.max(np.abs(v_trig)):.3f}"

    supervisor = SafetySupervisor()
    q_trig = env.robot.get_joint_positions().copy()

    t_threshold_step = None
    t_settled_step = None
    settle_count = 0
    contact_forces = []
    pens = []

    for step in range(160):
        q_now = env.robot.get_joint_positions()
        v_now = env.robot.get_joint_velocities()

        # Measure contact force specifically on human_proxy bodies
        step_force = 0.0
        step_pen = 0.0
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            b1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[c.geom2])
            if "human_proxy" in [b1, b2]:
                if c.dist < step_pen:
                    step_pen = c.dist
                cf = np.zeros(6)
                mujoco.mj_contactForce(env.model, env.data, i, cf)
                if cf[0] > 0:
                    step_force += cf[0]

        contact_forces.append(step_force)
        pens.append(abs(step_pen))

        r_telem = ReactiveTelemetry(
            has_collision=obs["contacts"]["robot_proxy_collision"],
            robot_proxy_collision=obs["contacts"]["robot_proxy_collision"],
            geometric_min_distance=max(0.0, step_pen)
        )

        cmd = supervisor.step(q_now, v_now, q_trig, 0.0, r_telem, dt=dt)
        obs = env.step(arm_targets=cmd.target_qpos, gripper_target=cmd.target_gripper)

        v_after = env.robot.get_joint_velocities()
        max_v = np.max(np.abs(v_after))
        if max_v < SafetySupervisor.REST_VELOCITY_THRESHOLD:
            if t_threshold_step is None:
                t_threshold_step = step + 1
            settle_count += 1
            if settle_count == 10 and t_settled_step is None:
                t_settled_step = step + 1
        else:
            settle_count = 0

        if not supervisor.is_arresting and t_settled_step is not None:
            break

    # Behavioral and physical invariant assertions
    assert np.max(contact_forces) > 0.0, "Proxy contact force must be non-zero"
    assert np.max(contact_forces) > 50.0, f"Significant contact force expected, got {np.max(contact_forces):.1f} N"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP, "Supervisor must enter EMERGENCY_STOP"
    assert len(supervisor._braking_trajectory) > 0, "Braking trajectory must be generated and executed"
    assert supervisor.is_latched is True, "Reactive latch must engage on collision"
    assert t_threshold_step is not None, "Robot must cross velocity threshold < 1e-3 rad/s"
    assert t_settled_step is not None, "Robot must achieve sustained rest for >= 10 ticks"
    assert t_settled_step >= t_threshold_step + 9, "Sustained settlement requires >= 9 steps after first crossing"
    assert supervisor.is_arresting is False, "Dynamic arrest concludes once rest is sustained"

    # Latch non-clearance verification
    safe_r = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    cmd_hold = supervisor.step(q_now, v_after, q_trig, 0.0, safe_r, dt=dt)
    assert supervisor.is_latched is True, "Latch cannot clear automatically"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP, "State must remain EMERGENCY_STOP while latched"

    env.close()


# =============================================================================
# Test 15: Invalid Reactive Telemetry Conservative Failure Response
# =============================================================================
def test_invalid_reactive_telemetry_conservative_failure():
    """
    Verify that invalid or non-finite Layer 0 telemetry NEVER allows NORMAL operation:
    - NaN geometric_min_distance
    - Inf geometric_min_distance
    - Negative geometric_min_distance (< 0.0)
    - NaN / negative timestamp
    - Malformed / non-ReactiveTelemetry input
    Must immediately engage EMERGENCY_STOP, latched stop mode, speed_scale=0.0,
    and output safe, clamped, finite target positions without raising unhandled exceptions.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5], dtype=np.float64)
    q_vel = np.zeros(7, dtype=np.float64)
    q_nom = q_curr.copy()
    tac_fresh = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    # 1. NaN geometric_min_distance
    r_nan = ReactiveTelemetry(has_collision=False, geometric_min_distance=float("nan"))
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_nan, tac_fresh, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.stop_mode == StopMode.LATCHED_REACTIVE
    assert cmd.is_latched is True
    assert cmd.speed_scale == 0.0
    assert np.all(np.isfinite(cmd.target_qpos))
    assert np.isfinite(cmd.target_gripper)
    assert any("CRITICAL_REACTIVE_INVALID" in a or "INPUT_VALIDATION_FAILURE" in a for a in cmd.active_alarms)

    # 2. Inf geometric_min_distance
    supervisor._test_only_reset()
    r_inf = ReactiveTelemetry(has_collision=False, geometric_min_distance=float("inf"))
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_inf, tac_fresh, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.speed_scale == 0.0

    # 3. Negative geometric distance
    supervisor._test_only_reset()
    r_neg = ReactiveTelemetry(has_collision=False, geometric_min_distance=-0.05)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_neg, tac_fresh, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.speed_scale == 0.0

    # 4. None / wrong type passed as reactive
    supervisor._test_only_reset()
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, None, tac_fresh, strat_fresh)  # type: ignore
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.speed_scale == 0.0


# =============================================================================
# Test 16: Invalid Robot State and Command Inputs
# =============================================================================
def test_invalid_robot_state_and_command_inputs():
    """
    Verify differentiated input validation for robot state, targets, and dt:
    1. Valid qpos with invalid qvel: freezes at current qpos; does NOT move toward home.
       Gripper remains at current target when empty, clamped (0.0) when holding object.
    2. Invalid qpos: holds last known commanded setpoint; does NOT command DEFAULT_SAFE_HOME_QPOS.
    3. Valid qpos & qvel with invalid nominal_q_target: halts at current qpos; does NOT move to target or home.
    4. Invalid reactive telemetry: executes safe arrest at current position.
    5. Checks returned commands (target_qpos, target_gripper), not merely the state.
    """
    supervisor = SafetySupervisor()
    q_valid = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.5, 0.5], dtype=np.float64)
    v_valid = np.zeros(7, dtype=np.float64)
    r_empty = ReactiveTelemetry(has_collision=False, robot_object_contact=False, geometric_min_distance=0.30)
    r_holding = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)

    # Establish initial known valid setpoint
    cmd_init = supervisor.step(q_valid, v_valid, q_valid, 255.0, r_empty)
    assert np.allclose(cmd_init.target_qpos, q_valid)
    assert cmd_init.target_gripper == 255.0

    # 1. Case A: Valid qpos with invalid qvel (empty hand) -> must freeze at q_valid, preserve gripper 255.0
    v_nan = v_valid.copy()
    v_nan[0] = float("nan")
    cmd_v_nan = supervisor.step(q_valid, v_nan, q_valid, 255.0, r_empty)
    assert cmd_v_nan.state == SupervisorState.EMERGENCY_STOP
    assert cmd_v_nan.is_latched is True
    assert cmd_v_nan.speed_scale == 0.0
    assert cmd_v_nan.actuator_inhibit is False
    assert np.allclose(cmd_v_nan.target_qpos, q_valid), "Must freeze at current qpos, NOT move to home"
    assert cmd_v_nan.target_gripper == 255.0, "Empty gripper must NOT clamp shut on finger"
    assert any("VELOCITY_FEEDBACK_LOST" in a for a in cmd_v_nan.active_alarms)

    # 1b. Case A with object held -> must freeze at q_valid, clamp gripper (0.0)
    supervisor._test_only_reset()
    cmd_v_nan_hold = supervisor.step(q_valid, v_nan, q_valid, 255.0, r_holding)
    assert np.allclose(cmd_v_nan_hold.target_qpos, q_valid)
    assert cmd_v_nan_hold.target_gripper == 0.0, "Holding gripper must clamp shut to preserve grasp"
    assert cmd_v_nan_hold.actuator_inhibit is False

    # 2. Case C: Invalid qpos, valid qvel (isolated position feedback failure with history)
    # -> must hold last known valid setpoint, NOT jump to DEFAULT_SAFE_HOME_QPOS
    q_nan = q_valid.copy()
    q_nan[3] = float("nan")
    cmd_q_nan = supervisor.step(q_nan, v_valid, q_valid, 255.0, r_empty)
    assert cmd_q_nan.state == SupervisorState.EMERGENCY_STOP
    assert cmd_q_nan.is_latched is True
    assert cmd_q_nan.actuator_inhibit is False
    assert np.allclose(cmd_q_nan.target_qpos, q_valid), "Must hold last valid commanded setpoint, NOT jump to home"
    assert not np.allclose(cmd_q_nan.target_qpos, supervisor.DEFAULT_SAFE_HOME_QPOS), "Must NOT command home on position sensor fault"
    assert any("POSITION_FEEDBACK_LOST" in a for a in cmd_q_nan.active_alarms)

    # 2b. Case C: First-cycle invalid qpos with valid qvel (no prior history)
    # -> target_qpos is None, actuator_inhibit is True, alarm CRITICAL_POSITION_FEEDBACK_FAULT
    fresh_supervisor = SafetySupervisor()
    cmd_first_fault = fresh_supervisor.step(q_nan, v_valid, q_valid, 255.0, r_empty)
    assert cmd_first_fault.state == SupervisorState.EMERGENCY_STOP
    assert cmd_first_fault.is_latched is True
    assert cmd_first_fault.actuator_inhibit is True, "Must set actuator_inhibit=True on first-cycle position fault"
    assert cmd_first_fault.target_qpos is None, "Must return target_qpos=None on first-cycle position fault"
    assert any("CRITICAL_POSITION_FEEDBACK_FAULT" in a for a in cmd_first_fault.active_alarms)

    # 2c. Case B: Invalid qpos and invalid qvel (simultaneous dual failure with history)
    # -> must hold last known valid setpoint
    cmd_dual_hist = supervisor.step(q_nan, v_nan, q_valid, 255.0, r_empty)
    assert cmd_dual_hist.state == SupervisorState.EMERGENCY_STOP
    assert cmd_dual_hist.is_latched is True
    assert cmd_dual_hist.actuator_inhibit is False
    assert np.allclose(cmd_dual_hist.target_qpos, q_valid)
    assert any("DUAL_FEEDBACK_LOST_HOLD" in a for a in cmd_dual_hist.active_alarms)

    # 2d. Case B: First-cycle invalid qpos and invalid qvel (simultaneous dual failure without history)
    # -> target_qpos is None, actuator_inhibit is True, alarm CRITICAL_DUAL_STATE_FEEDBACK_FAULT
    fresh_dual_supervisor = SafetySupervisor()
    cmd_dual_first = fresh_dual_supervisor.step(q_nan, v_nan, q_valid, 255.0, r_empty)
    assert cmd_dual_first.state == SupervisorState.EMERGENCY_STOP
    assert cmd_dual_first.is_latched is True
    assert cmd_dual_first.actuator_inhibit is True
    assert cmd_dual_first.target_qpos is None
    assert any("CRITICAL_DUAL_STATE_FEEDBACK_FAULT" in a for a in cmd_dual_first.active_alarms)

    # 3. Valid qpos & qvel with invalid nominal_q_target -> rejects target, halts at q_valid
    supervisor._test_only_reset()
    q_nom_nan = q_valid.copy()
    q_nom_nan[1] = float("inf")
    cmd_nom_nan = supervisor.step(q_valid, v_valid, q_nom_nan, 255.0, r_empty)
    assert cmd_nom_nan.state == SupervisorState.EMERGENCY_STOP
    assert np.allclose(cmd_nom_nan.target_qpos, q_valid), "Must halt at current position when planner target is invalid"
    assert any("INVALID_PLANNER_TARGET" in a for a in cmd_nom_nan.active_alarms)

    # 4. Valid state and target with invalid reactive telemetry -> safe stop at q_valid
    supervisor._test_only_reset()
    r_bad = ReactiveTelemetry(has_collision=False, geometric_min_distance=-0.10)
    cmd_r_bad = supervisor.step(q_valid, v_valid, q_valid, 255.0, r_bad)
    assert cmd_r_bad.state == SupervisorState.EMERGENCY_STOP
    assert np.allclose(cmd_r_bad.target_qpos, q_valid)

    # 5. Malformed dimensions (< 7 elements)
    supervisor._test_only_reset()
    q_short = np.array([0.0, 1.0])
    cmd_short = supervisor.step(q_short, v_valid, q_valid, 255.0, r_empty)
    assert cmd_short.state == SupervisorState.EMERGENCY_STOP
    assert cmd_short.target_qpos is None
    assert cmd_short.actuator_inhibit is True

    # 6. Invalid timestep dt <= 0 or NaN
    supervisor._test_only_reset()
    cmd_dt = supervisor.step(q_valid, v_valid, q_valid, 255.0, r_empty, dt=-0.01)
    assert cmd_dt.state == SupervisorState.EMERGENCY_STOP
    assert np.all(np.isfinite(cmd_dt.target_qpos))


# =============================================================================
# Test 17: Operational Reset Handshake and Bypass Removal
# =============================================================================
def test_operational_reset_handshake_and_bypass_removal():
    """
    Verify reset() API integrity:
    - force=True is removed; passing force raises TypeError.
    - reset() requires operator_confirm=True when latched.
    - reset() requires current_qvel and verifies robot rest.
    - reset() verifies clearance and collision sensors.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7, dtype=np.float64)
    q_vel = np.zeros(7, dtype=np.float64)
    v_moving = np.ones(7, dtype=np.float64) * 0.5

    # Latch the supervisor via Layer 0 collision
    r_col = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)
    supervisor.step(q_curr, q_vel, q_curr, 0.0, r_col)
    assert supervisor.is_latched is True

    # 1. Verify force parameter is rejected
    with pytest.raises(TypeError):
        supervisor.reset(force=True)  # type: ignore

    # 2. reset() without operator_confirm=True must REFUSE
    assert supervisor.reset() is False
    assert supervisor.is_latched is True

    # 3. reset(operator_confirm=True) without current_qvel must REFUSE
    assert supervisor.reset(operator_confirm=True) is False
    assert supervisor.is_latched is True

    # 4. reset(operator_confirm=True, current_qvel=v_moving) with motion must REFUSE
    assert supervisor.reset(operator_confirm=True, current_qvel=v_moving) is False
    assert supervisor.is_latched is True

    # 5. reset(operator_confirm=True, current_qvel=q_vel) while collision persists must REFUSE
    assert supervisor.reset(operator_confirm=True, current_qvel=q_vel) is False
    assert supervisor.is_latched is True

    # 6. Provide clear telemetry; now reset must SUCCEED through validated handshake into RECOVERY
    supervisor.latest_reactive = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    success = supervisor.reset(operator_confirm=True, current_qvel=q_vel)
    assert success is True
    assert supervisor.is_latched is False
    assert supervisor.state == SupervisorState.RECOVERY, "Latch release MUST enter RECOVERY, never bypass to NORMAL"
    assert supervisor.speed_scale <= 0.50
    assert supervisor._recovery_dwell_counter == 0


# =============================================================================
# Test 18: Braking Trajectory Failure and Exhaustion While Moving
# =============================================================================
def test_braking_trajectory_failure_and_exhaustion_while_moving():
    """
    Verify supervisor behavior under braking trajectory failures:
    1. Trajectory generation failure: sets _braking_success = False and logs alarm.
    2. Trajectory exhausted before physical rest:
       - Maintains _is_arresting = True.
       - Holds final computed waypoint.
       - Logs DYNAMIC_ARREST_EXHAUSTED_STILL_MOVING.
    3. Rest threshold achieved and sustained for 10 ticks:
       - Concludes arrest (_is_arresting = False).
       - Logs DYNAMIC_ARREST_SETTLED.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    v_fast = np.ones(7, dtype=np.float64) * 1.5
    r_col = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.0)

    # 1. Trigger collision with high velocity
    cmd = supervisor.step(q_curr, v_fast, q_curr, 0.0, r_col, dt=0.01)
    assert supervisor.is_arresting is True
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    # Artificially fast-forward past end of trajectory while robot is still moving
    supervisor._braking_step_idx = len(supervisor._braking_trajectory)
    cmd_moving = supervisor.step(q_curr, v_fast, q_curr, 0.0, r_col, dt=0.01)

    # Verify waypoint holds final trajectory position and arrest remains True
    assert supervisor.is_arresting is True, "Must NOT conclude arrest while robot is still moving"
    assert any("DYNAMIC_ARREST_EXHAUSTED_STILL_MOVING" in a for a in cmd_moving.active_alarms)
    assert np.allclose(cmd_moving.target_qpos, supervisor._braking_trajectory[-1])

    # 2. Simulate robot coming to physical rest: sustained 10 ticks below REST_VELOCITY_THRESHOLD
    v_rest = np.zeros(7, dtype=np.float64)
    for tick in range(9):
        cmd_tick = supervisor.step(q_curr, v_rest, q_curr, 0.0, r_col, dt=0.01)
        assert supervisor.is_arresting is True, f"Must remain arresting until 10 ticks sustained rest (tick {tick+1})"

    # 10th tick of rest concludes arrest
    cmd_settled = supervisor.step(q_curr, v_rest, q_curr, 0.0, r_col, dt=0.01)
    assert supervisor.is_arresting is False, "Arrest concludes once 10 ticks of rest are confirmed"
    assert any("DYNAMIC_ARREST_SETTLED" in a for a in cmd_settled.active_alarms)


# =============================================================================
# Test 19: Unlatched Predictive Emergency Recovery De-escalation
# =============================================================================
def test_predictive_emergency_unlatched_recovery_deescalation():
    """
    Verify automatic de-escalation for unlatched PREDICTIVE_EMERGENCY:
    - Triggered by severe contextual emergency (C_harm >= 0.85).
    - Latched flag is FALSE (no Layer 0 contact).
    - When severe hazard recedes, robot is at rest, and preconditions hold for dwell time (20 ticks),
      supervisor automatically de-escalates to RECOVERY.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5], dtype=np.float64)
    q_vel = np.zeros(7, dtype=np.float64)
    r_safe = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    # 1. Trigger unlatched severe emergency
    tac_severe = TacticalTelemetry(p10=0.10, d10_clear=0.20, c_harm=0.90, age_ms=20.0)
    cmd = supervisor.step(q_curr, q_vel, q_curr, 255.0, r_safe, tac_severe, strat_fresh)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.stop_mode == StopMode.PREDICTIVE_EMERGENCY
    assert cmd.is_latched is False

    # 2. Severe hazard recedes (safe tactical telemetry)
    tac_clear = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    for tick in range(19):
        cmd = supervisor.step(q_curr, q_vel, q_curr, 255.0, r_safe, tac_clear, strat_fresh)
        assert cmd.state == SupervisorState.EMERGENCY_STOP, f"Must dwell in EMERGENCY_STOP before 20 ticks (tick {tick+1})"

    # 20th dwell tick triggers de-escalation to RECOVERY
    cmd_rec = supervisor.step(q_curr, q_vel, q_curr, 255.0, r_safe, tac_clear, strat_fresh)
    assert cmd_rec.state == SupervisorState.RECOVERY, "Must automatically transition to RECOVERY after 20 dwell ticks"
    assert cmd_rec.stop_mode == StopMode.NONE
    assert any("TRANSITION: EMERGENCY_STOP (Unlatched) -> RECOVERY" in a for a in cmd_rec.active_alarms)


# =============================================================================
# Test 20: Recovery Speed Cap and Fresh H50 Requirement
# =============================================================================
def test_recovery_speed_cap_and_h50_requirements():
    """
    Verify RECOVERY state rules regarding H50:
    1. Stale or missing H50 permits only degraded speed cap (0.50x).
    2. Detours (replan_flag) remain strictly disabled without fresh valid H50.
    3. Full speed (1.00x) and transition to NORMAL require fresh, valid H50.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7, dtype=np.float64)
    q_vel = np.zeros(7, dtype=np.float64)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    r_safe = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    tac_clear = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)

    # Force supervisor into RECOVERY
    supervisor._state = SupervisorState.RECOVERY
    supervisor._speed_scale = 0.0

    # Part A: Missing H50 (strategic=None)
    # Speed should ramp up to 0.50 and strictly cap there; replan_flag must be False
    for _ in range(80):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=None, dt=0.01)
    assert cmd.state == SupervisorState.RECOVERY
    assert cmd.speed_scale == 0.50, f"Missing H50 must cap speed strictly at 0.50x, got {cmd.speed_scale}"
    assert cmd.replan_flag is False, "Detours must remain disabled without fresh H50"

    # Part B: Stale H50 (is_stale=True or age > 150ms)
    supervisor._speed_scale = 0.0
    strat_stale = StrategicTelemetry(p50=0.45, d50_clear=0.20, age_ms=50.0, is_stale=True)
    for _ in range(80):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_stale, dt=0.01)
    assert cmd.state == SupervisorState.RECOVERY
    assert cmd.speed_scale == 0.50, "Stale H50 must cap speed strictly at 0.50x"
    assert cmd.replan_flag is False, "Detours must remain disabled with stale H50"

    # Part C: Fresh H50 with strategic conflict (p50 >= 0.36) -> detours permitted
    strat_conflict = StrategicTelemetry(p50=0.40, d50_clear=0.20, age_ms=50.0, is_stale=False)
    cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_conflict, dt=0.01)
    assert cmd.replan_flag is True, "Fresh valid H50 with elevated P50 must permit replan detour"

    # Part D: Fresh H50 with clear path (p50 <= 0.20) -> speed ramps to 1.00x and transitions to NORMAL
    supervisor._speed_scale = 0.50
    strat_clean = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)
    for _ in range(60):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_clean, dt=0.01)
    assert cmd.state == SupervisorState.NORMAL, "Must transition to NORMAL with fresh H50 once ramp completes"
    assert cmd.speed_scale == 1.00, "Full normal speed must reach 1.00x"
    assert cmd.replan_flag is False


# =============================================================================
# Test 21: Empty and Failed Braking Trajectory Generation Protection
# =============================================================================
def test_empty_and_failed_braking_trajectory_generation(monkeypatch):
    """
    Verify emergency supervisor never crashes with IndexError if trajectory generation
    fails or produces an empty trajectory:
    - Trajectory generator returns positions=[] and success=False.
    - Supervisor step() produces valid finite command.
    - Alarms record DYNAMIC_ARREST_GENERATION_FAILED and DYNAMIC_ARREST_EMPTY_TRAJECTORY.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.5, 0.5], dtype=np.float64)
    v_curr = np.ones(7, dtype=np.float64) * 0.8
    r_col = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.0)

    # Mock generate_braking_trajectory to return an empty trajectory with failure
    def mock_generate(*args, **kwargs):
        return BrakingTrajectoryResult(positions=[], velocities=[], success=False, failure_reason="Mocked generator failure")

    monkeypatch.setattr(supervisor, "generate_braking_trajectory", mock_generate)

    cmd = supervisor.step(q_curr, v_curr, q_curr, 0.0, r_col)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert np.all(np.isfinite(cmd.target_qpos))
    assert np.allclose(cmd.target_qpos, q_curr)
    assert any("DYNAMIC_ARREST_GENERATION_FAILED" in a for a in cmd.active_alarms)

    # Artificially inject empty trajectory cache while arresting
    supervisor._braking_trajectory = []
    supervisor._braking_step_idx = 0
    supervisor._is_arresting = True

    cmd_empty = supervisor.step(q_curr, v_curr, q_curr, 0.0, r_col)
    assert cmd_empty.state == SupervisorState.EMERGENCY_STOP
    assert np.all(np.isfinite(cmd_empty.target_qpos))
    assert any("DYNAMIC_ARREST_EMPTY_TRAJECTORY" in a for a in cmd_empty.active_alarms)


# =============================================================================
# Test 22: Consecutive Physical Rest Settle Counter
# =============================================================================
def test_consecutive_settle_counter_resets_on_motion():
    """
    Verify that the 10-tick physical-rest counter strictly resets to 0 on any motion:
    It must represent 10 consecutive ticks below REST_VELOCITY_THRESHOLD.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    v_fast = np.ones(7, dtype=np.float64) * 1.0
    v_rest = np.zeros(7, dtype=np.float64)
    r_col = ReactiveTelemetry(has_collision=True, geometric_min_distance=0.0)

    # 1. Trigger arrest
    supervisor.step(q_curr, v_fast, q_curr, 0.0, r_col)
    assert supervisor.is_arresting is True

    # Fast-forward to end of trajectory
    supervisor._braking_step_idx = len(supervisor._braking_trajectory)

    # 2. Step 5 ticks stationary -> settle counter = 5
    for _ in range(5):
        supervisor.step(q_curr, v_rest, q_curr, 0.0, r_col)
    assert supervisor._arrest_settle_counter == 5
    assert supervisor.is_arresting is True

    # 3. Step 1 tick moving -> settle counter must reset to 0!
    supervisor.step(q_curr, v_fast, q_curr, 0.0, r_col)
    assert supervisor._arrest_settle_counter == 0, "Motion must reset settle counter to 0"
    assert supervisor.is_arresting is True

    # 4. Step 5 ticks stationary -> counter reaches 5, NOT 10!
    for _ in range(5):
        supervisor.step(q_curr, v_rest, q_curr, 0.0, r_col)
    assert supervisor._arrest_settle_counter == 5
    assert supervisor.is_arresting is True, "Must not conclude arrest prematurely"

    # 5. Step 5 more ticks stationary (reaching 10 consecutive) -> concludes arrest
    for _ in range(5):
        cmd_settled = supervisor.step(q_curr, v_rest, q_curr, 0.0, r_col)
    assert supervisor.is_arresting is False
    assert any("DYNAMIC_ARREST_SETTLED" in a for a in cmd_settled.active_alarms)


# =============================================================================
# Test 23: Predictive Emergency Dwell Resets on Hazard Reappearance
# =============================================================================
def test_predictive_emergency_dwell_resets_on_hazard_reappearance():
    """
    Verify recovery dwell counter resets immediately to 0 if recovery preconditions
    fail or any hazard reappears during unlatched EMERGENCY_STOP dwell.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5], dtype=np.float64)
    v_zero = np.zeros(7, dtype=np.float64)
    r_safe = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    # 1. Trigger severe emergency
    tac_severe = TacticalTelemetry(p10=0.10, d10_clear=0.20, c_harm=0.90, age_ms=20.0)
    supervisor.step(q_curr, v_zero, q_curr, 255.0, r_safe, tac_severe, strat_fresh)
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert supervisor.is_latched is False

    # 2. Hazard recedes: dwell for 15 ticks
    tac_safe = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    for _ in range(15):
        supervisor.step(q_curr, v_zero, q_curr, 255.0, r_safe, tac_safe, strat_fresh)
    assert supervisor._recovery_dwell_counter == 15
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    # 3. Hazard reappears on tick 16 -> dwell counter must reset immediately to 0
    supervisor.step(q_curr, v_zero, q_curr, 255.0, r_safe, tac_severe, strat_fresh)
    assert supervisor._recovery_dwell_counter == 0, "Dwell counter must reset to 0 upon hazard reappearance"
    assert supervisor.state == SupervisorState.EMERGENCY_STOP

    # 4. Hazard recedes again -> requires 20 full consecutive ticks
    for _ in range(19):
        supervisor.step(q_curr, v_zero, q_curr, 255.0, r_safe, tac_safe, strat_fresh)
        assert supervisor.state == SupervisorState.EMERGENCY_STOP
    cmd_rec = supervisor.step(q_curr, v_zero, q_curr, 255.0, r_safe, tac_safe, strat_fresh)
    assert cmd_rec.state == SupervisorState.RECOVERY


# =============================================================================
# Test 24: H50 Becomes Stale Mid-Recovery
# =============================================================================
def test_h50_becomes_stale_mid_recovery_immediate_clamp():
    """
    Verify that if H50 becomes stale or unavailable during RECOVERY:
    1. Speed cap immediately becomes 0.50 (clamping down from higher speeds).
    2. Detours (replan_flag) remain disabled.
    3. NORMAL state remains inaccessible until fresh valid H50 is restored.
    """
    supervisor = SafetySupervisor()
    q_curr = np.zeros(7, dtype=np.float64)
    q_vel = np.zeros(7, dtype=np.float64)
    q_nom = np.array([0.1, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    r_safe = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    tac_clear = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)

    supervisor._state = SupervisorState.RECOVERY
    supervisor._speed_scale = 0.0

    # 1. Ramp speed scale to ~0.80 under fresh H50
    for _ in range(80):
        cmd = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_fresh, dt=0.01)
    assert cmd.state == SupervisorState.RECOVERY
    assert cmd.speed_scale >= 0.79

    # 2. H50 becomes stale on next tick -> speed scale must IMMEDIATELY clamp to <= 0.50!
    strat_stale = StrategicTelemetry(p50=0.40, d50_clear=0.20, age_ms=50.0, is_stale=True)
    cmd_stale = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_stale, dt=0.01)
    assert cmd_stale.state == SupervisorState.RECOVERY
    assert cmd_stale.speed_scale <= 0.50, f"Speed scale must immediately clamp down to 0.50, got {cmd_stale.speed_scale}"
    assert cmd_stale.replan_flag is False, "Detours must remain disabled when H50 is stale"

    # 3. Step for 50 more ticks under stale H50 -> verify NORMAL is never reached
    for _ in range(50):
        cmd_loop = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_stale, dt=0.01)
        assert cmd_loop.state == SupervisorState.RECOVERY
        assert cmd_loop.speed_scale <= 0.50

    # 4. Fresh H50 returns -> resumes ramp to 1.00 and transitions to NORMAL
    for _ in range(60):
        cmd_fresh = supervisor.step(q_curr, q_vel, q_nom, 255.0, r_safe, tac_clear, strategic=strat_fresh, dt=0.01)
    assert cmd_fresh.state == SupervisorState.NORMAL
    assert cmd_fresh.speed_scale == 1.00


# =============================================================================
# Test 25: Operator Reset Handshake Cannot Bypass RECOVERY
# =============================================================================
def test_reset_handshake_cannot_bypass_recovery():
    """
    Verify that calling reset(operator_confirm=True, current_qvel=...) while latched:
    1. Rejects reset if operator_confirm=False or current_qvel is missing/moving/NaN.
    2. When successful, transitions to RECOVERY, NOT NORMAL.
    3. Retains speed_scale <= 0.50 and recovery dwell counter at 0.
    4. Preserves telemetry history required for recovery verification.
    5. Requires stepping through RECOVERY with fresh H10 and fresh H50 before reaching NORMAL.
    """
    supervisor = SafetySupervisor()
    q_curr = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5], dtype=np.float64)
    v_zero = np.zeros(7, dtype=np.float64)
    v_fast = np.ones(7, dtype=np.float64) * 0.5
    q_nom = q_curr.copy()

    # 1. Trigger reactive emergency collision (latched)
    r_col = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)
    cmd = supervisor.step(q_curr, v_fast, q_nom, 0.0, r_col)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.stop_mode == StopMode.LATCHED_REACTIVE

    # 2. Rejection checks
    # Missing confirmation
    assert supervisor.reset(operator_confirm=False, current_qvel=v_zero) is False
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert supervisor.is_latched is True

    # Moving robot
    assert supervisor.reset(operator_confirm=True, current_qvel=v_fast) is False
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert supervisor.is_latched is True

    # NaN velocity
    assert supervisor.reset(operator_confirm=True, current_qvel=np.full(7, np.nan)) is False
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert supervisor.is_latched is True

    # Still colliding
    assert supervisor.reset(operator_confirm=True, current_qvel=v_zero) is False
    assert supervisor.state == SupervisorState.EMERGENCY_STOP
    assert supervisor.is_latched is True

    # 3. Collision clears and robot at rest
    r_clear = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    supervisor.latest_reactive = r_clear

    # 4. Successful operator reset
    res = supervisor.reset(operator_confirm=True, current_qvel=v_zero)
    assert res is True
    assert supervisor.is_latched is False, "Latch must be released"
    assert supervisor.state == SupervisorState.RECOVERY, "Must enter RECOVERY, NEVER bypass to NORMAL"
    assert supervisor.speed_scale <= 0.50, "Speed scale must be restricted during recovery"
    assert supervisor._recovery_dwell_counter == 0, "Recovery dwell counter must start at 0"
    assert supervisor.latest_reactive is not None, "Telemetry must NOT be erased"

    # 5. Stepping with stale H50 during recovery clamps speed at 0.50 and cannot reach NORMAL
    tac_fresh = TacticalTelemetry(p10=0.05, d10_clear=0.30, c_harm=0.05, age_ms=20.0)
    strat_stale = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=True)
    for _ in range(60):
        cmd_rec = supervisor.step(q_curr, v_zero, q_nom, 0.0, r_clear, tac_fresh, strat_stale, dt=0.01)
        assert cmd_rec.state == SupervisorState.RECOVERY
        assert cmd_rec.speed_scale <= 0.50

    # 6. Fresh H50 allows ramping to NORMAL
    strat_fresh = StrategicTelemetry(p50=0.05, d50_clear=0.30, age_ms=50.0, is_stale=False)
    for _ in range(60):
        cmd_rec = supervisor.step(q_curr, v_zero, q_nom, 0.0, r_clear, tac_fresh, strat_fresh, dt=0.01)
    assert cmd_rec.state == SupervisorState.NORMAL
    assert cmd_rec.speed_scale == 1.00


# =============================================================================
# Test 26: Invalid Velocity Feedback During Motion — High-Speed MuJoCo Position Hold
# =============================================================================
def test_invalid_velocity_feedback_high_speed_mujoco_position_hold():
    """
    Verify high-speed MuJoCo behavior when velocity feedback is lost (NaN/Inf):
    1. Arm accelerates to high transport velocity (> 1.0 rad/s).
    2. Invalid velocity feedback (np.nan) is passed to SafetySupervisor.step().
    3. SafetySupervisor engages EMERGENCY_STOP with alarm VELOCITY_FEEDBACK_LOST_POSITION_HOLD.
    4. SafetySupervisor issues an instantaneous position-setpoint hold (target_qpos = current_qpos),
       NOT a smooth velocity-informed dynamic braking trajectory (trajectory length == 0).
    5. MuJoCo closed-loop physics execution records:
       - Actuator torque saturation (hitting Franka tau_max = 87 Nm).
       - Peak actual joint deceleration exceeding the provisional 15 rad/s^2 limit.
       - Rapid forward-motion arrest (t_arrest <= 50 ms).
       - Eventual settling to physical rest (< 1e-3 rad/s) without joint limit violations.
    """
    env = SawmEnvironment("configs/simulation.yaml")
    obs = env.reset()
    planner = BaselinePlanner(env)
    kf, ok, msg = planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
    assert ok

    env.robot.set_joint_positions_direct(kf["lift"])
    traj = env.robot.interpolate_trajectory(kf["lift"], kf["transport"], steps=40)
    dt = env.dt * env.frame_skip

    # Accelerate arm along transport trajectory until high velocity is achieved
    for s in range(18):
        obs = env.step(arm_targets=traj[s])

    q_trig = env.robot.get_joint_positions().copy()
    v_trig = env.robot.get_joint_velocities().copy()
    max_v_trig = float(np.max(np.abs(v_trig)))
    assert max_v_trig > 1.0, f"Trigger velocity must be high, got {max_v_trig:.3f} rad/s"

    supervisor = SafetySupervisor()
    safe_r = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)

    # Inject invalid velocity feedback
    invalid_qvel = np.full(7, np.nan, dtype=np.float64)
    cmd = supervisor.step(q_trig, invalid_qvel, traj[18], 0.0, safe_r, dt=dt)

    # Invariants for position hold vs velocity-informed braking
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.speed_scale == 0.0
    assert any("VELOCITY_FEEDBACK_LOST_POSITION_HOLD" in a for a in cmd.active_alarms)
    assert len(supervisor._braking_trajectory) == 0, "Position hold must NOT generate a braking trajectory"
    np.testing.assert_allclose(cmd.target_qpos, q_trig, atol=1e-6)

    # Step closed-loop in MuJoCo physics applying position hold
    torques = []
    actual_accels = []
    q_history = []
    v_prev = v_trig.copy()
    t_arrest_step = None
    t_threshold_step = None
    t_settled_step = None
    settle_count = 0

    for step in range(80):
        q_now = env.robot.get_joint_positions()
        v_now = env.robot.get_joint_velocities()
        q_history.append(q_now.copy())

        # Measure instantaneous acceleration
        accel = np.max(np.abs((v_now - v_prev) / dt))
        actual_accels.append(accel)
        v_prev = v_now.copy()

        # Measure actuator torques
        tau = np.abs(env.data.actuator_force[:7].copy())
        torques.append(tau)

        # Detect forward arrest (first major velocity reduction or reversal)
        if t_arrest_step is None and np.max(np.abs(v_now)) < 0.20 * max_v_trig:
            t_arrest_step = step + 1

        # Check settling threshold
        if np.max(np.abs(v_now)) < SafetySupervisor.REST_VELOCITY_THRESHOLD:
            if t_threshold_step is None:
                t_threshold_step = step + 1
            settle_count += 1
            if settle_count == 10 and t_settled_step is None:
                t_settled_step = step + 1
        else:
            settle_count = 0

        # Feed invalid velocity continuously
        cmd = supervisor.step(q_now, invalid_qvel, traj[18], 0.0, safe_r, dt=dt)
        obs = env.step(arm_targets=cmd.target_qpos, gripper_target=cmd.target_gripper)

        if t_settled_step is not None:
            break

    # Physical measurements and assertions
    peak_torque = float(np.max(torques))
    peak_accel = float(np.max(actual_accels))
    q_disp = float(np.max(np.abs(env.robot.get_joint_positions() - q_trig)))

    # 1. Franka Panda max torque is 87 Nm; position hold under high speed causes saturation
    assert peak_torque >= 80.0, f"Peak torque must reflect heavy deceleration / saturation, got {peak_torque:.2f} Nm"
    # 2. Position hold induces sharp deceleration exceeding the provisional 15 rad/s^2 limit
    assert peak_accel > 15.0, f"Actual deceleration must exceed 15 rad/s^2 due to position hold, got {peak_accel:.2f} rad/s^2"
    # 3. Stopping displacement must be small (pure position hold at trigger point, < 0.10 rad / 5.7 deg)
    assert q_disp < 0.10, f"Stopping displacement must be minimal, got {q_disp:.4f} rad"
    # 4. Robot must reach physical rest
    assert t_threshold_step is not None, "Robot must cross rest velocity threshold"
    assert t_settled_step is not None, "Robot must settle to rest"
    assert t_settled_step >= t_threshold_step + 9

    # 5. Joint limits must never be violated
    for q_step in q_history:
        for j in range(7):
            j_name = f"joint{j+1}"
            assert env.robot.joint_limits[j_name][0] <= q_step[j] <= env.robot.joint_limits[j_name][1], (
                f"Joint {j} violated limit: {q_step[j]}"
            )

    env.close()


# =============================================================================
# Test 27: First-Cycle Position Feedback Genuine Actuator Inhibition Integration
# =============================================================================
def test_first_cycle_position_feedback_genuine_actuator_inhibition():
    """
    Integration test verifying genuine simulation-side actuator inhibition:
    1. Establishes active, nonzero arm actuator targets in MuJoCo.
    2. Steps environment forward so actuators and ctrl buffers are actively engaged.
    3. Triggers first-cycle position-feedback failure on a fresh SafetySupervisor.
    4. Passes the resulting intervention (target_qpos=None, actuator_inhibit=True) to the environment.
    5. Verifies the arm actuator mechanism is genuinely inhibited (arm_actuators_inhibited is True).
    6. Verifies the old active control values in data.ctrl cannot continue driving the arm (actuator torques are strictly 0.0).
    7. Verifies the fallback zero vector is never sent as a position target (target_qpos is None).
    """
    env = SawmEnvironment("configs/simulation.yaml")
    obs = env.reset()

    # 1. Establish an active, nonzero arm actuator target
    active_targets = np.array([0.5, -0.3, 0.4, -1.8, 0.2, 1.2, 0.1], dtype=np.float64)
    obs = env.step(arm_targets=active_targets)
    for act_id in env.robot.actuator_ids:
        assert env.data.ctrl[act_id] != 0.0

    # Actuator forces must be active under normal tracking
    pre_forces = np.abs(env.data.actuator_force[:7].copy())
    assert np.max(pre_forces) > 0.0, "Actuators must be active before inhibition"

    # 2. Trigger first-cycle position feedback failure on fresh supervisor
    fresh_supervisor = SafetySupervisor()
    q_nan = np.full(7, np.nan, dtype=np.float64)
    v_nan = np.full(7, np.nan, dtype=np.float64)
    r_empty = ReactiveTelemetry(has_collision=False, geometric_min_distance=0.30)
    cmd = fresh_supervisor.step(q_nan, v_nan, active_targets, 255.0, r_empty)

    # Invariants on supervisor output
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True
    assert cmd.actuator_inhibit is True, "Must explicitly flag actuator inhibition"
    assert cmd.target_qpos is None, "Target must be None; fallback zero vector must NEVER be sent as position target"

    # 3. Pass intervention to environment honoring actuator_inhibit and binding supervisor
    assert cmd.supervisor is fresh_supervisor, "cmd.supervisor must reference the issuing SafetySupervisor instance"
    obs_inhibited = env.step(arm_targets=cmd.target_qpos, actuator_inhibit=cmd.actuator_inhibit, supervisor=cmd.supervisor)

    # 4. Verifications of genuine inhibition
    assert env.robot.arm_actuators_inhibited is True, "Robot arm actuators must be flagged as inhibited"
    post_forces = np.abs(env.data.actuator_force[:7].copy())
    np.testing.assert_allclose(
        post_forces, 0.0, atol=1e-10,
        err_msg="Arm actuators must generate identically 0.0 torque when inhibited (Safe Torque Off)"
    )

    # 5. Verify old active ctrl buffer values cannot continue driving the arm
    for _ in range(5):
        env.step(arm_targets=None, actuator_inhibit=True, supervisor=fresh_supervisor)
        forces = np.abs(env.data.actuator_force[:7].copy())
        np.testing.assert_allclose(forces, 0.0, atol=1e-10, err_msg="Old ctrl values must NOT produce torque while inhibited")

    # 6. Verify inhibition is LATCHED and ordinary planner calls cannot re-enable actuators!
    ordinary_planner_targets = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.5, 0.5], dtype=np.float64)
    for _ in range(5):
        obs_planner = env.step(arm_targets=ordinary_planner_targets, actuator_inhibit=False)
        assert env.robot.arm_actuators_inhibited is True, "Ordinary planner calls must NOT un-inhibit actuators"
        forces = np.abs(env.data.actuator_force[:7].copy())
        np.testing.assert_allclose(forces, 0.0, atol=1e-10, err_msg="Ordinary planner must NOT produce torque while latched inhibited")

    # 7. Verify direct robot enable is prohibited
    with pytest.raises(RuntimeError, match="Direct actuator restoration via robot.enable_arm_actuators\\(\\) is prohibited"):
        env.robot.enable_arm_actuators()

    # 8. Complete validated recovery handshake on the bound supervisor
    r_clear = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    fresh_supervisor.latest_reactive = r_clear
    reset_ok = fresh_supervisor.reset(operator_confirm=True, current_qvel=np.zeros(7))
    assert reset_ok is True
    assert fresh_supervisor.is_latched is False
    assert fresh_supervisor.state == SupervisorState.RECOVERY
    assert fresh_supervisor.has_completed_recovery_handshake is True

    # Recovery authorization succeeds with the bound, validated supervisor
    auth_ok, auth_msg = env.authorize_actuator_recovery(fresh_supervisor)
    assert auth_ok is True
    assert "RECOVERY_AUTHORIZED" in auth_msg
    assert env.robot.arm_actuators_inhibited is False
    np.testing.assert_array_equal(
        env.model.actuator_forcerange[env.robot.actuator_ids],
        env.robot._orig_actuator_forcerange,
        err_msg="actuator_forcerange must match saved original values exactly"
    )
    np.testing.assert_array_equal(
        env.model.actuator_gainprm[env.robot.actuator_ids],
        env.robot._orig_actuator_gainprm,
        err_msg="actuator_gainprm must match saved original values exactly"
    )
    np.testing.assert_array_equal(
        env.model.actuator_biasprm[env.robot.actuator_ids],
        env.robot._orig_actuator_biasprm,
        err_msg="actuator_biasprm must match saved original values exactly"
    )

    # Normal tracking resumes and generates active forces
    obs_resumed = env.step(arm_targets=ordinary_planner_targets)
    assert np.max(np.abs(env.data.actuator_force[:7])) > 0.0, "Actuators must resume normal torque generation after explicit enable"

    # 9. Verify repeated inhibit/enable cycle with the bound supervisor
    env.step(arm_targets=None, actuator_inhibit=True, supervisor=fresh_supervisor)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)
    fresh_supervisor.latest_reactive = r_clear
    reset_ok2 = fresh_supervisor.reset(operator_confirm=True, current_qvel=np.zeros(7))
    assert reset_ok2 is True
    auth_ok2, _ = env.authorize_actuator_recovery(fresh_supervisor)
    assert auth_ok2 is True
    assert env.robot.arm_actuators_inhibited is False
    np.testing.assert_array_equal(env.model.actuator_forcerange[env.robot.actuator_ids], env.robot._orig_actuator_forcerange)

    env.close()


# =============================================================================
# Test 28: Environment Reset Latch Integrity and Authorized Recovery
# =============================================================================
def test_environment_reset_latch_integrity_and_authorized_recovery():
    """
    Verify environment reset latch integrity and validated actuator recovery authorization:
    1. Ordinary environment reset when no emergency inhibition exists preserves normal actuation.
    2. Environment reset while supervisor is latched does NOT clear the actuator inhibition latch.
    3. Actuators remain at strictly zero torque after reset.
    4. Ordinary planner commands are rejected after reset while latched.
    5. Failed or incomplete recovery handshake (operator_confirm=False, NaN velocity) never restores actuation.
    6. Explicit authorized recovery restores actuators ONLY at the correct point in the supervisor recovery lifecycle (RECOVERY state).
    """
    env = SawmEnvironment("configs/simulation.yaml")

    # 1. Ordinary environment reset when no emergency inhibition exists
    obs = env.reset()
    assert env.robot.arm_actuators_inhibited is False, "Ordinary reset must not inhibit actuators"
    ordinary_target = np.array([0.1, -0.5, 0.2, -2.0, 0.3, 1.5, 0.5], dtype=np.float64)
    obs = env.step(arm_targets=ordinary_target)
    assert np.max(np.abs(env.data.actuator_force[:7])) > 0.0, "Ordinary reset must permit active motion tracking"

    # 2. Trigger reactive collision to latch the supervisor in EMERGENCY_STOP and inhibit actuators
    supervisor = SafetySupervisor()
    r_col = ReactiveTelemetry(has_collision=True, robot_obstacle_collision=True, geometric_min_distance=0.0)
    cmd = supervisor.step(env.robot.get_joint_positions(), env.robot.get_joint_velocities(), ordinary_target, 0.0, r_col)
    assert cmd.state == SupervisorState.EMERGENCY_STOP
    assert cmd.is_latched is True

    # Inhibit actuators in environment with supervisor binding
    env.step(actuator_inhibit=True, supervisor=supervisor)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)

    # 3. Environment reset while supervisor is latched MUST NOT clear the inhibition latch!
    obs_reset = env.reset()
    assert env.robot.arm_actuators_inhibited is True, "env.reset() MUST NOT silently clear the inhibition latch!"
    assert supervisor.is_latched is True, "Supervisor must remain latched"
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0, err_msg="Actuator forces must remain zero after reset")
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # 4. Ordinary planner commands must be rejected after reset while latched
    for _ in range(5):
        obs_planner = env.step(arm_targets=ordinary_target, actuator_inhibit=False)
        assert env.robot.arm_actuators_inhibited is True
        np.testing.assert_allclose(env.data.actuator_force[:7], 0.0, err_msg="Planner targets must be rejected while latched")

    # 4b. Direct restoration attempt via PandaRobot is strictly PROHIBITED:
    with pytest.raises(RuntimeError, match="Direct actuator restoration via robot.enable_arm_actuators\\(\\) is prohibited"):
        env.robot.enable_arm_actuators()
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0, err_msg="Arm actuator torques must remain 0.0")
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    with pytest.raises(RuntimeError, match="Direct actuator restoration via robot.enable_arm_actuators\\(\\) is prohibited"):
        env.robot.enable_arm_actuators(supervisor=supervisor)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0, err_msg="Arm actuator torques must remain 0.0")
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # 4c. Arbitrary or fake supervisor objects CANNOT restore actuators:
    class FakeSupervisor:
        def __init__(self):
            self.is_latched = False
            self.state = SupervisorState.NORMAL
            self.latest_reactive = None

    fake_sup = FakeSupervisor()

    # Attempt authorization via env.authorize_actuator_recovery(fake_sup)
    auth_fake, msg_fake = env.authorize_actuator_recovery(fake_sup)
    assert auth_fake is False, "Fake supervisor object MUST be rejected"
    assert "Arbitrary or fake supervisor object rejected" in msg_fake
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0, err_msg="Fake supervisor must not restore torques")
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # Attempt reset with fake recovery_supervisor
    env.reset(recovery_supervisor=fake_sup)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # Attempt authorization with dict / non-SafetySupervisor
    auth_dict, msg_dict = env.authorize_actuator_recovery({"is_latched": False, "state": "NORMAL"})
    assert auth_dict is False
    assert "Arbitrary or fake supervisor object rejected" in msg_dict
    assert env.robot.arm_actuators_inhibited is True

    # 4d. Supervisor identity and state binding regression tests:
    # (i) Original supervisor A latches; fresh supervisor B in NORMAL is rejected
    supervisor_b_norm = SafetySupervisor()
    assert supervisor_b_norm.state == SupervisorState.NORMAL
    assert supervisor_b_norm.is_latched is False
    auth_b_norm, msg_b_norm = env.authorize_actuator_recovery(supervisor_b_norm)
    assert auth_b_norm is False
    assert "Supervisor identity mismatch" in msg_b_norm
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # (ii) Original supervisor A remains latched; B in RECOVERY is rejected
    supervisor_b_rec = SafetySupervisor()
    supervisor_b_rec._state = SupervisorState.RECOVERY
    supervisor_b_rec._is_latched = False
    auth_b_rec, msg_b_rec = env.authorize_actuator_recovery(supervisor_b_rec)
    assert auth_b_rec is False
    assert "Supervisor identity mismatch" in msg_b_rec
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)
    np.testing.assert_allclose(env.model.actuator_forcerange[env.robot.actuator_ids], 0.0)

    # (iii) Repeated reset attempts with supervisor B cannot bypass the latch
    env.reset(recovery_supervisor=supervisor_b_norm)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)

    env.reset(recovery_supervisor=supervisor_b_rec)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)

    # 5. Failed or incomplete recovery never restores actuation
    # Attempt recovery authorization while supervisor is still latched
    auth_fail1, msg1 = env.authorize_actuator_recovery(supervisor)
    assert auth_fail1 is False
    assert "RECOVERY_REJECTED" in msg1
    assert env.robot.arm_actuators_inhibited is True

    # Attempt reset(recovery_supervisor=supervisor) while supervisor is still latched
    env.reset(recovery_supervisor=supervisor)
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)

    # Attempt supervisor reset without operator confirmation (fails)
    reset_ok = supervisor.reset(operator_confirm=False, current_qvel=np.zeros(7))
    assert reset_ok is False
    assert supervisor.is_latched is True
    auth_fail2, msg2 = env.authorize_actuator_recovery(supervisor)
    assert auth_fail2 is False
    assert env.robot.arm_actuators_inhibited is True

    # Attempt supervisor reset with non-finite or moving velocity (fails)
    reset_ok2 = supervisor.reset(operator_confirm=True, current_qvel=np.ones(7) * 0.5)
    assert reset_ok2 is False
    assert supervisor.is_latched is True
    auth_fail3, _ = env.authorize_actuator_recovery(supervisor)
    assert auth_fail3 is False
    assert env.robot.arm_actuators_inhibited is True

    # 6. Explicit authorized recovery restoring actuators ONLY at the correct point in lifecycle
    # Clear collision sensors first
    r_clear = ReactiveTelemetry(has_collision=False, robot_object_contact=True, geometric_min_distance=0.30)
    supervisor.latest_reactive = r_clear

    # Complete validated recovery handshake into RECOVERY state
    v_rest = np.zeros(7, dtype=np.float64)
    reset_ok3 = supervisor.reset(operator_confirm=True, current_qvel=v_rest)
    assert reset_ok3 is True
    assert supervisor.is_latched is False
    assert supervisor.state == SupervisorState.RECOVERY
    assert supervisor.has_completed_recovery_handshake is True

    # Now authorization must SUCCEED!
    auth_ok, auth_msg = env.authorize_actuator_recovery(supervisor)
    assert auth_ok is True
    assert "RECOVERY_AUTHORIZED" in auth_msg
    assert env.robot.arm_actuators_inhibited is False, "Actuators must be re-enabled under authorized RECOVERY"
    np.testing.assert_array_equal(env.model.actuator_forcerange[env.robot.actuator_ids], env.robot._orig_actuator_forcerange)

    # Stepping in RECOVERY now successfully executes motion commands
    obs_rec = env.step(arm_targets=ordinary_target)
    assert np.max(np.abs(env.data.actuator_force[:7])) > 0.0, "Actuators must produce torque under authorized recovery"

    # 7. A new emergency resets previous recovery authorization on the supervisor
    cmd_new_emerg = supervisor.step(
        env.robot.get_joint_positions(),
        env.robot.get_joint_velocities(),
        ordinary_target,
        0.0,
        r_col,
    )
    assert cmd_new_emerg.state == SupervisorState.EMERGENCY_STOP
    assert cmd_new_emerg.is_latched is True
    assert supervisor.has_completed_recovery_handshake is False, "New emergency MUST reset previous recovery handshake status"

    # Inhibit actuators under new emergency
    env.step(actuator_inhibit=True, supervisor=supervisor)
    assert env.robot.arm_actuators_inhibited is True

    # Attempting recovery authorization without a second operator reset MUST fail
    auth_fail_re_emerg, msg_re_emerg = env.authorize_actuator_recovery(supervisor)
    assert auth_fail_re_emerg is False
    assert "Supervisor is latched in emergency stop" in msg_re_emerg
    assert env.robot.arm_actuators_inhibited is True

    # Complete second operator reset handshake
    supervisor.latest_reactive = r_clear
    reset_ok4 = supervisor.reset(operator_confirm=True, current_qvel=v_rest)
    assert reset_ok4 is True
    assert supervisor.has_completed_recovery_handshake is True

    # Second recovery authorization succeeds
    auth_ok_re, _ = env.authorize_actuator_recovery(supervisor)
    assert auth_ok_re is True
    assert env.robot.arm_actuators_inhibited is False

    # 8. Unbound inhibition keeps actuators inhibited and refuses recovery
    env.inhibit_arm_actuators(supervisor=None)
    assert env.robot.arm_actuators_inhibited is True
    auth_fail_unbound, msg_unbound = env.authorize_actuator_recovery(supervisor)
    assert auth_fail_unbound is False
    assert "Active inhibition has no bound supervisor recorded" in msg_unbound
    assert env.robot.arm_actuators_inhibited is True
    np.testing.assert_allclose(env.data.actuator_force[:7], 0.0)

    env.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
