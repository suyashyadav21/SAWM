"""
Safety Supervisor State Machine for SAWM (Milestone 6).

Implements the centralized, deterministic finite-state machine (FSM) that arbitrates:
1. Layer 0 Reactive Telemetry (100 Hz nominal): MuJoCo contact pairs, collision flags, geometric distances.
2. Layer 1 Tactical Predictive Telemetry (25 Hz design target): H=10 collision probability, clearance, contextual harm.
3. Layer 2 Strategic Predictive Telemetry (10 Hz nominal): H=50 corridor conflict analysis and detour validation.

States:
- NORMAL: Nominal operation (s_v = 1.0), nominal acceleration limits.
- CAUTION: Moderate risk or strategic conflict (s_v = 0.50 - 0.80, default 0.60).
- SLOWDOWN: Primary predictive arrest with bounded deceleration.
- EMERGENCY_STOP: Full arrest (s_v = 0.0), latched on Layer 0 contact, unlatched on severe contextual emergency or reflex.
- RECOVERY: Preconditions verified, smooth speed ramp (s_v: 0.0 -> 1.0 at 1.0/s).

CRITICAL QUALIFICATIONS & CONSTRAINTS:
1. Post-Contact Simulated Emergency-Arrest Scope: In the tested scenarios, the reactive supervisor
   functions strictly as a post-contact simulated emergency-arrest mechanism. It does NOT provide
   collision prevention, human injury prevention, certified safety, or universal stopping guarantees.
2. Provisional Grasp-Preservation Limit: 15.0 rad/s² is an unvalidated design limit subject
   to empirical validation via physical object-slip experiments. It is NOT universally enforced
   under dynamic multi-joint coupling or reactive dynamic arrest where torques saturate at +-87 Nm.
3. Kinematic Stopping Distance Formula: q_stop = q + sign(q_dot) * q_dot^2 / (2 * a_brake) is a
   simplified 1D kinematic estimate. Predictive braking is implemented as a bounded trajectory
   accounting for initial velocity, actuator torque limits (87 Nm / 12 Nm), gravity bias,
   joint limits, and payload.
4. Actuator Dynamics vs Kinematic Setpoints: The diagonal inertia and gravity-bias approximations
   are strictly provisional and do not constitute verified actuator-level braking. Actual MuJoCo
   dynamic response depends on position servo gains (kp in [2000, 4500], kd in [200, 450]), torque
   saturation, and payload forces. Planned trajectories represent kinematic setpoints.
5. Latched Reactive Stop: When moving at non-zero velocity upon Layer 0 contact, the arm executes
   an active dynamic arrest phase before achieving steady-state position hold; it is NOT assumed
   that commanding the frozen position instantly stops the arm.
6. Simulation Contact Measurements vs Injury Predictions: Measured contact normal forces in MuJoCo
   (e.g. 338.61 N against human-proxy capsule, 357.01 N against obstacle, 643.50 N against table)
   with peak decelerations up to 108-150 rad/s² are simulation measurements reflecting high-gain
   position servo stiffness and rigid body contact physics, NOT validated human injury predictions.
7. Separate Stopping Time Definitions:
   - t_arrest (Initial forward-motion arrest): Time until forward joint velocity reverses sign or drops
     below 5% of trigger value (80-100 ms under torque saturation).
   - t_threshold (First residual velocity threshold crossing): First timestep where max |qdot| < 1e-3 rad/s
     (600-650 ms in reactive obstacle arrest, 720 ms in smooth slowdown).
   - t_settled (Sustained physical rest): Time from which max |qdot| remains strictly below 1e-3 rad/s
     for >= 10 consecutive ticks (810 ms in smooth slowdown, 1090-1280 ms in reactive collision, > 1.0s under table impact).
   The theoretical 120 ms target was an idealized kinematic estimate (v0 / a_brake), not an established
   dynamic physical capability.
8. Soft Real-Time Timing: All timing claims (e.g. 25 Hz / 40 ms cycle) are provisional pending
   complete 1,000-cycle benchmarks.
9. Position Hold vs Velocity-Informed Dynamic Braking: Commanding current_qpos when current_qvel
   is invalid is strictly an instantaneous position-setpoint hold, NOT velocity-informed controlled dynamic braking.
   In high-speed test configurations (such as the evaluated 1.43 rad/s transport corridor maneuver with payload),
   commanding an instantaneous position hold produced actuator torque saturation (+-87 Nm) and a peak measured joint
   deceleration of 153.18 rad/s² exceeding the provisional 15 rad/s² design limit. The measured 153.18 rad/s² is an empirical
   result from the tested high-speed corridor configuration, NOT a universal maximum, theoretical bound, or stopping
   guarantee across other speeds, payloads, or configurations. No controlled dynamic braking or grasp preservation guarantee
   is claimed when velocity feedback is unavailable.
10. First-Cycle Position Feedback Fault: If current_qpos is invalid on the first control cycle before any setpoint
    has been commanded, no trustworthy physical position is available. The supervisor returns target_qpos=None and
    sets actuator_inhibit=True with alarm CRITICAL_POSITION_FEEDBACK_FAULT. This condition does NOT constitute a safe
    physical hold. Callers must honor actuator_inhibit=True to genuinely disable/inhibit arm actuators (Safe Torque Off),
    preventing any fallback position targets or previous ctrl buffer values from driving the arm.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SupervisorState(str, Enum):
    """Safety Supervisor state machine operational states."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    SLOWDOWN = "SLOWDOWN"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY = "RECOVERY"


class StopMode(str, Enum):
    """Taxonomy of stopping mechanisms."""
    NONE = "NONE"
    CONTROLLED_DECEL = "CONTROLLED_DECEL"          # SLOWDOWN: path-constrained smooth deceleration
    PREDICTIVE_EMERGENCY = "PREDICTIVE_EMERGENCY"  # EMERGENCY_STOP (Unlatched): reflex halt
    LATCHED_REACTIVE = "LATCHED_REACTIVE"          # EMERGENCY_STOP (Latched): physical contact arrest


@dataclass
class ReactiveTelemetry:
    """Layer 0 reactive telemetry (100 Hz nominal)."""
    has_collision: bool = False
    robot_obstacle_collision: bool = False
    robot_proxy_collision: bool = False
    robot_table_collision: bool = False
    robot_object_contact: bool = False
    geometric_min_distance: float = 999.0
    timestamp: float = 0.0

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate reactive telemetry fields."""
        errors = []
        if np.isnan(self.geometric_min_distance) or np.isinf(self.geometric_min_distance):
            errors.append("Invalid geometric_min_distance (NaN or Inf)")
        elif self.geometric_min_distance < 0.0:
            errors.append(f"Negative geometric_min_distance ({self.geometric_min_distance})")
        if np.isnan(self.timestamp) or np.isinf(self.timestamp) or self.timestamp < 0.0:
            errors.append("Invalid timestamp")
        return (len(errors) == 0), errors


@dataclass
class TacticalTelemetry:
    """Layer 1 tactical predictive telemetry (25 Hz design target, 40 ms period)."""
    p10: float = 0.0                   # Collision probability at H=10 [0.0, 1.0]
    d10_clear: float = 999.0           # Minimum clearance at H=10 in meters (>= 0.0)
    c_harm: float = 0.0                # Contextual harm score [0.0, 1.0]
    closure_rate: float = 0.0          # Relative closing speed in m/s (negative = closing)
    human_proxy_distance: float = 999.0# Distance to dynamic human proxy in meters
    timestamp: float = 0.0             # Timestamp of generation
    age_ms: float = 0.0                # Age relative to evaluation tick

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate telemetry values, ranges, and finiteness."""
        errors = []
        for name, val in [("p10", self.p10), ("d10_clear", self.d10_clear),
                          ("c_harm", self.c_harm), ("closure_rate", self.closure_rate),
                          ("human_proxy_distance", self.human_proxy_distance),
                          ("timestamp", self.timestamp), ("age_ms", self.age_ms)]:
            if np.isnan(val) or np.isinf(val):
                errors.append(f"Non-finite value in {name}: {val}")
        if not (0.0 <= self.p10 <= 1.0):
            errors.append(f"p10 out of valid probability range [0, 1]: {self.p10}")
        if not (0.0 <= self.c_harm <= 1.0):
            errors.append(f"c_harm out of valid range [0, 1]: {self.c_harm}")
        if self.d10_clear < 0.0:
            errors.append(f"d10_clear cannot be negative: {self.d10_clear}")
        if self.human_proxy_distance < 0.0:
            errors.append(f"human_proxy_distance cannot be negative: {self.human_proxy_distance}")
        if self.age_ms < 0.0:
            errors.append(f"age_ms cannot be negative: {self.age_ms}")
        return (len(errors) == 0), errors


@dataclass
class StrategicTelemetry:
    """Layer 2 strategic predictive telemetry (10 Hz nominal, 100 ms period)."""
    p50: float = 0.0                   # Collision probability at H=50 [0.0, 1.0]
    d50_clear: float = 999.0           # Minimum clearance at H=50 in meters (>= 0.0)
    timestamp: float = 0.0             # Timestamp of generation
    age_ms: float = 0.0                # Age relative to evaluation tick
    is_stale: bool = False             # True if age > 150 ms

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate strategic telemetry values and ranges."""
        errors = []
        for name, val in [("p50", self.p50), ("d50_clear", self.d50_clear),
                          ("timestamp", self.timestamp), ("age_ms", self.age_ms)]:
            if np.isnan(val) or np.isinf(val):
                errors.append(f"Non-finite value in {name}: {val}")
        if not (0.0 <= self.p50 <= 1.0):
            errors.append(f"p50 out of valid range [0, 1]: {self.p50}")
        if self.d50_clear < 0.0:
            errors.append(f"d50_clear cannot be negative: {self.d50_clear}")
        if self.age_ms < 0.0:
            errors.append(f"age_ms cannot be negative: {self.age_ms}")
        return (len(errors) == 0), errors


@dataclass
class BrakingTrajectoryResult:
    """Detailed outcome of bounded predictive braking trajectory calculation."""
    positions: List[np.ndarray]
    velocities: List[np.ndarray]
    success: bool
    failure_reason: Optional[str] = None
    stopping_steps: int = 0
    estimated_stopping_time_s: float = 0.0


@dataclass
class InterventionCommand:
    """Command produced by Safety Supervisor at each 100 Hz control tick."""
    state: SupervisorState
    speed_scale: float                           # s_v in [0.0, 1.0]
    target_qpos: Optional[np.ndarray]            # (7,) commanded arm joint targets, or None if actuator inhibit / position unavailable
    target_gripper: float                        # commanded gripper target (0.0 to 255.0)
    replan_flag: bool                            # True if strategic replan is recommended
    is_latched: bool                             # True if supervisor is in latched reactive stop
    stop_mode: StopMode
    active_alarms: List[str] = field(default_factory=list)
    actuator_inhibit: bool = False               # True if arm actuators must be inhibited (Safe Torque Off / power cutoff)
    supervisor: Optional[Any] = None             # Reference to issuing SafetySupervisor instance (simulation-level binding)


class SafetySupervisor:
    """
    Centralized deterministic Safety Supervisor FSM for Franka Panda robot.
    """

    # Hardware limits for Franka Emika Panda (7 arm joints)
    JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    JOINT_LIMITS_UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float64)
    TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)  # Nm
    ESTIMATED_JOINT_INERTIA = np.array([2.5, 3.0, 1.5, 1.5, 0.5, 0.5, 0.2], dtype=np.float64)  # kg*m^2

    # Provisional Grasp-Preservation Limit:
    # PROVISIONAL: 15.0 rad/s² is an unvalidated design limit subject to empirical validation
    # via physical object-slip experiments.
    MAX_JOINT_ACCEL_PROVISIONAL: float = 15.0  # rad/s²
    REST_VELOCITY_THRESHOLD: float = 1e-3      # rad/s
    JOINT_LIMIT_SAFETY_MARGIN_RAD: float = 0.087  # ~5 degrees in radians
    DEFAULT_SAFE_HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)

    # Tactical Risk Thresholds
    P10_ALARM_THRESHOLD: float = 0.36
    P10_CRITICAL_THRESHOLD: float = 0.70
    P10_RECOVERY_THRESHOLD: float = 0.20

    D10_CAUTION_THRESHOLD: float = 0.20       # m (Normal -> Caution)
    D10_RECOVERY_CLEARANCE: float = 0.25      # m (Caution -> Normal hysteresis)
    D10_MODERATE_THRESHOLD: float = 0.08      # m (Caution -> Slowdown)
    D10_CRITICAL_THRESHOLD: float = 0.03      # m (Critical Fast-Path)
    D10_DYNAMIC_APPROACH_DIST: float = 0.06   # m (Dynamic Fast-Path distance)
    CLOSURE_RATE_CRITICAL: float = -0.60      # m/s (Dynamic Fast-Path closure)

    # Contextual Harm Thresholds
    C_HARM_MODERATE_THRESHOLD: float = 0.30   # Contextual moderate warning
    C_HARM_CRITICAL_THRESHOLD: float = 0.70   # Contextual critical fast-path (SLOWDOWN)
    C_HARM_SEVERE_EMERGENCY: float = 0.85     # Severe contextual emergency (EMERGENCY_STOP)
    HUMAN_PROXY_EMERGENCY_DIST: float = 0.03  # m (Severe human proximity emergency)

    # Strategic Risk Thresholds
    P50_STRATEGIC_THRESHOLD: float = 0.36
    P50_RECOVERY_THRESHOLD: float = 0.20
    D50_RECOVERY_THRESHOLD: float = 0.15      # m

    # Layer 0 Reactive Thresholds
    L0_CONTACT_DISTANCE_THRESHOLD: float = 0.01  # m (Reactive distance breach)
    L0_RECOVERY_DISTANCE_THRESHOLD: float = 0.15 # m (Layer 0 recovery clearance)

    # Freshness Boundaries
    H10_FRESH_MAX_MS: float = 40.0
    H10_STALE_MAX_MS: float = 80.0
    H50_FRESH_MAX_MS: float = 150.0
    H50_STALE_MAX_MS: float = 300.0

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # FSM state
        self._state: SupervisorState = SupervisorState.NORMAL
        self._is_latched: bool = False
        self._is_arresting: bool = False
        self._stop_mode: StopMode = StopMode.NONE

        # Speed scaling factor
        self._speed_scale: float = 1.0

        # Latched frozen reference
        self._frozen_qpos: Optional[np.ndarray] = None

        # Recovery handshake completion tracker (simulation-level binding)
        self._has_completed_operator_recovery: bool = False

        # Active dynamic braking trajectory cache
        self._braking_trajectory: List[np.ndarray] = []
        self._braking_step_idx: int = 0
        self._braking_success: bool = True

        # Separate 2-of-3 filter buffers: physical collision vs contextual warning
        self._physical_risk_history: List[bool] = [False, False, False]
        self._contextual_risk_history: List[bool] = [False, False, False]

        # Dwell confirmation counters
        self._recovery_dwell_counter: int = 0
        self._clear_path_dwell_counter: int = 0
        self._arrest_settle_counter: int = 0

        # Latest received telemetries
        self.latest_reactive: Optional[ReactiveTelemetry] = None
        self.latest_tactical: Optional[TacticalTelemetry] = None
        self.latest_strategic: Optional[StrategicTelemetry] = None

        # Last known valid commanded positions and gripper target
        self._last_valid_cmd_qpos: Optional[np.ndarray] = None
        self._last_valid_gripper_target: float = 255.0

        # Active alarm logs
        self._active_alarms: List[str] = []

    # -------------------------------------------------------------------------
    # Properties & State Inspection
    # -------------------------------------------------------------------------
    @property
    def state(self) -> SupervisorState:
        return self._state

    @property
    def is_latched(self) -> bool:
        return self._is_latched

    @property
    def is_arresting(self) -> bool:
        return self._is_arresting

    @property
    def stop_mode(self) -> StopMode:
        return self._stop_mode

    @property
    def speed_scale(self) -> float:
        return self._speed_scale

    @property
    def active_alarms(self) -> List[str]:
        return list(self._active_alarms)

    @property
    def has_completed_recovery_handshake(self) -> bool:
        """
        True if supervisor has completed an explicit operator-confirmed recovery handshake.
        Simulation-level recovery verification mechanism; not a security boundary against
        arbitrary Python code execution.
        """
        return self._has_completed_operator_recovery

    def reset(
        self,
        operator_confirm: bool = False,
        current_qvel: Optional[np.ndarray] = None
    ) -> bool:
        """
        Reset supervisor to initial state.

        CRITICAL LATCH INTEGRITY:
        If the supervisor is currently in a latched emergency stop, calling reset()
        without explicit operator confirmation and valid physical clearance will
        REFUSE to clear the latch! All operational latch clearing routes through
        clear_latched_emergency_stop() ensuring collision, velocity, and clearance
        checks cannot be bypassed.
        """
        if self._is_latched:
            if not operator_confirm:
                self._active_alarms.append(
                    "RESET_REFUSED: Active reactive emergency latch cannot be cleared without explicit operator confirmation"
                )
                return False

            if current_qvel is None:
                self._active_alarms.append(
                    "RESET_REFUSED: Active reactive emergency latch requires current_qvel verification to ensure physical rest"
                )
                return False

            success, msg = self.clear_latched_emergency_stop(current_qvel, operator_confirm=operator_confirm)
            if not success:
                self._active_alarms.append(f"RESET_REFUSED: {msg}")
                return False

            # Latch clearance enters RECOVERY. Do NOT bypass to NORMAL or erase telemetry!
            # The supervisor remains in RECOVERY with speed_scale <= 0.50 until recovery
            # preconditions and H10/H50 freshness requirements allow ramping to NORMAL.
            return True

        self._state = SupervisorState.NORMAL
        self._is_latched = False
        self._is_arresting = False
        self._stop_mode = StopMode.NONE
        self._speed_scale = 1.0
        self._frozen_qpos = None
        self._braking_trajectory = []
        self._braking_step_idx = 0
        self._braking_success = True
        self._physical_risk_history = [False, False, False]
        self._contextual_risk_history = [False, False, False]
        self._recovery_dwell_counter = 0
        self._clear_path_dwell_counter = 0
        self._arrest_settle_counter = 0
        self._last_valid_cmd_qpos = None
        self._last_valid_gripper_target = 255.0
        self.latest_reactive = None
        self.latest_tactical = None
        self.latest_strategic = None
        self._active_alarms = []
        return True

    def _test_only_reset(self) -> None:
        """
        Internal test-only reset fixture to reinitialize state without validation checks.
        DO NOT USE IN PRODUCTION OR OPERATIONAL CODE.
        """
        self._state = SupervisorState.NORMAL
        self._is_latched = False
        self._is_arresting = False
        self._stop_mode = StopMode.NONE
        self._speed_scale = 1.0
        self._frozen_qpos = None
        self._braking_trajectory = []
        self._braking_step_idx = 0
        self._braking_success = True
        self._physical_risk_history = [False, False, False]
        self._contextual_risk_history = [False, False, False]
        self._recovery_dwell_counter = 0
        self._clear_path_dwell_counter = 0
        self._arrest_settle_counter = 0
        self._last_valid_cmd_qpos = None
        self._last_valid_gripper_target = 255.0
        self.latest_reactive = None
        self.latest_tactical = None
        self.latest_strategic = None
        self._active_alarms = []

    # -------------------------------------------------------------------------
    # Simplified Kinematics & Bounded Braking Trajectory Generation
    # -------------------------------------------------------------------------
    @classmethod
    def compute_stopping_distance_1d(
        cls,
        q: np.ndarray,
        q_dot: np.ndarray,
        a_brake: float = MAX_JOINT_ACCEL_PROVISIONAL
    ) -> np.ndarray:
        """
        Compute simplified 1D kinematic stopping position estimate:
            q_stop,i = q_i + sign(q_dot_i) * q_dot_i^2 / (2 * a_brake)
        """
        if a_brake <= 0:
            raise ValueError(f"Braking deceleration magnitude must be strictly positive, got {a_brake}")
        q = np.asarray(q, dtype=np.float64)
        q_dot = np.asarray(q_dot, dtype=np.float64)
        if np.any(np.isnan(q)) or np.any(np.isinf(q)):
            raise ValueError("Non-finite values in position q")
        if np.any(np.isnan(q_dot)) or np.any(np.isinf(q_dot)):
            raise ValueError("Non-finite values in velocity q_dot")

        stopping_disp = np.sign(q_dot) * (q_dot ** 2) / (2.0 * a_brake)
        return q + stopping_disp

    @classmethod
    def generate_braking_trajectory(
        cls,
        q_start: np.ndarray,
        q_dot_start: np.ndarray,
        gravity_torques: Optional[np.ndarray] = None,
        effective_inertia: Optional[np.ndarray] = None,
        payload_mass: float = 0.1,
        dt: float = 0.01,
        a_max_provisional: float = MAX_JOINT_ACCEL_PROVISIONAL,
        max_steps: int = 200,
    ) -> BrakingTrajectoryResult:
        """
        Generate bounded predictive braking trajectory accounting for torque limits,
        gravity bias, joint limits with safety margin, and payload inertia.
        """
        q_curr = np.asarray(q_start, dtype=np.float64).copy()
        q_dot_curr = np.asarray(q_dot_start, dtype=np.float64).copy()

        # Input validation
        if np.any(np.isnan(q_curr)) or np.any(np.isinf(q_curr)):
            return BrakingTrajectoryResult([], [], False, "Non-finite values in q_start")
        if np.any(np.isnan(q_dot_curr)) or np.any(np.isinf(q_dot_curr)):
            return BrakingTrajectoryResult([], [], False, "Non-finite values in q_dot_start")
        if dt <= 0:
            return BrakingTrajectoryResult([], [], False, f"Invalid timestep dt <= 0: {dt}")
        if a_max_provisional <= 0:
            return BrakingTrajectoryResult([], [], False, f"Invalid a_max_provisional <= 0: {a_max_provisional}")

        # Clamp initial position to hardware limits with safety margin
        q_min_arr = cls.JOINT_LIMITS_LOWER + cls.JOINT_LIMIT_SAFETY_MARGIN_RAD
        q_max_arr = cls.JOINT_LIMITS_UPPER - cls.JOINT_LIMIT_SAFETY_MARGIN_RAD
        q_curr = np.clip(q_curr, q_min_arr, q_max_arr)

        if effective_inertia is None:
            inertia = cls.ESTIMATED_JOINT_INERTIA.copy()
        else:
            inertia = np.asarray(effective_inertia, dtype=np.float64).copy()
        if np.any(inertia <= 0):
            return BrakingTrajectoryResult([], [], False, f"Inertia elements must be strictly positive: {inertia}")

        payload_inertia = max(0.0, payload_mass) * (0.8 ** 2)
        inertia += payload_inertia

        if gravity_torques is None:
            tau_bias = np.array([0.0, 5.0, 0.0, 4.0, 0.0, 2.5, 0.0], dtype=np.float64)
        else:
            tau_bias = np.asarray(gravity_torques, dtype=np.float64)
            if np.any(np.isnan(tau_bias)) or np.any(np.isinf(tau_bias)):
                return BrakingTrajectoryResult([], [], False, "Non-finite values in gravity_torques")

        sign_vel = np.sign(q_dot_curr)
        tau_avail = cls.TORQUE_LIMITS - (sign_vel * tau_bias)
        tau_avail = np.maximum(tau_avail, 1.0)

        a_torque_max = tau_avail / inertia
        a_decel = np.minimum(a_max_provisional, a_torque_max)

        positions = [q_curr.copy()]
        velocities = [q_dot_curr.copy()]

        step = 0
        while np.max(np.abs(q_dot_curr)) > cls.REST_VELOCITY_THRESHOLD and step < max_steps:
            for i in range(7):
                v = q_dot_curr[i]
                if abs(v) <= cls.REST_VELOCITY_THRESHOLD:
                    q_dot_curr[i] = 0.0
                else:
                    decel = a_decel[i]
                    if v > 0:
                        v_next = max(0.0, v - decel * dt)
                    else:
                        v_next = min(0.0, v + decel * dt)
                    q_dot_curr[i] = v_next

                q_next = q_curr[i] + q_dot_curr[i] * dt
                q_curr[i] = float(np.clip(q_next, q_min_arr[i], q_max_arr[i]))

            positions.append(q_curr.copy())
            velocities.append(q_dot_curr.copy())
            step += 1

        final_max_vel = float(np.max(np.abs(q_dot_curr)))
        success = (final_max_vel <= cls.REST_VELOCITY_THRESHOLD)
        reason = None if success else f"Iteration limit ({max_steps} steps) reached before rest (residual vel: {final_max_vel:.4f} rad/s)"

        return BrakingTrajectoryResult(
            positions=positions,
            velocities=velocities,
            success=success,
            failure_reason=reason,
            stopping_steps=step,
            estimated_stopping_time_s=step * dt,
        )

    # -------------------------------------------------------------------------
    # Tactical Alarm Evaluation Logic (Regions & Filter)
    # -------------------------------------------------------------------------
    def evaluate_tactical_alarms(
        self,
        tactical: Optional[TacticalTelemetry]
    ) -> Tuple[bool, bool, bool, bool, List[str]]:
        """
        Evaluate Tactical Alarms according to Section 3 & 5 of M6 Specification:
        1. Severe Emergency (Filter Bypassed, 0 ms delay):
           (C_harm >= 0.85) OR (human_proxy_distance <= 0.03m)
        2. Tier 2 Critical Fast-Path (Filter bypassed, 0 ms delay):
           (P10 >= 0.70) OR (d10 <= 0.03m) OR (d10 <= 0.06m AND closure_rate <= -0.60m/s) OR (0.70 <= C_harm < 0.85)
        3. Tier 1 Physical Moderate Risk (Subject to 2-of-3 filter at 25 Hz):
           (0.36 <= P10 < 0.70) OR (0.03m < d10 <= 0.08m)
        4. Tier 1 Contextual Moderate Warning (Subject to 2-of-3 filter at 25 Hz):
           (0.30 <= C_harm < 0.70)

        Returns:
            severe_emergency (bool)
            critical_fast_path (bool)
            physical_mod_confirmed (bool)
            contextual_mod_confirmed (bool)
            alarm_messages (List[str])
        """
        alarms = []
        if tactical is None:
            alarms.append("TACTICAL_TELEMETRY_MISSING: Missing H10 telemetry forces conservative response")
            return False, False, False, False, alarms

        valid, val_errs = tactical.validate()
        if not valid:
            for err in val_errs:
                alarms.append(f"CRITICAL_TELEMETRY_INVALID: {err}")
            return True, True, False, False, alarms

        # Check for expired H10
        if tactical.age_ms > self.H10_STALE_MAX_MS:
            alarms.append(f"TACTICAL_TELEMETRY_EXPIRED: age={tactical.age_ms:.1f}ms > {self.H10_STALE_MAX_MS}ms")

        p10 = tactical.p10
        d10 = tactical.d10_clear
        c_harm = tactical.c_harm
        closure = tactical.closure_rate
        proxy_dist = tactical.human_proxy_distance

        # Severe Emergency: C_harm >= 0.85 or human proxy <= 0.03m
        severe_harm = (c_harm >= self.C_HARM_SEVERE_EMERGENCY)
        severe_proxy = (proxy_dist <= self.HUMAN_PROXY_EMERGENCY_DIST)
        severe_emergency = severe_harm or severe_proxy

        if severe_emergency:
            if severe_harm:
                alarms.append(f"SEVERE_CONTEXTUAL_EMERGENCY: C_harm={c_harm:.2f} >= {self.C_HARM_SEVERE_EMERGENCY}")
            if severe_proxy:
                alarms.append(f"SEVERE_HUMAN_PROXIMITY_EMERGENCY: proxy_dist={proxy_dist*1000:.1f}mm <= {self.HUMAN_PROXY_EMERGENCY_DIST*1000:.0f}mm")

        # Tier 2 Critical Fast-Path Checks
        crit_p10 = (p10 >= self.P10_CRITICAL_THRESHOLD)
        crit_d10 = (d10 <= self.D10_CRITICAL_THRESHOLD)
        crit_approach = (d10 <= self.D10_DYNAMIC_APPROACH_DIST) and (closure <= self.CLOSURE_RATE_CRITICAL)
        crit_harm = (self.C_HARM_CRITICAL_THRESHOLD <= c_harm < self.C_HARM_SEVERE_EMERGENCY)

        critical_fast_path = severe_emergency or crit_p10 or crit_d10 or crit_approach or crit_harm

        if critical_fast_path and not severe_emergency:
            if crit_p10:
                alarms.append(f"CRITICAL_FAST_PATH: High collision prob P10={p10:.3f} >= {self.P10_CRITICAL_THRESHOLD}")
            if crit_d10:
                alarms.append(f"CRITICAL_FAST_PATH: Severe proximity d10={d10*1000:.1f}mm <= {self.D10_CRITICAL_THRESHOLD*1000:.0f}mm")
            if crit_approach:
                alarms.append(f"CRITICAL_FAST_PATH: Rapid closure={closure:.2f}m/s with d10={d10*1000:.1f}mm")
            if crit_harm:
                alarms.append(f"CRITICAL_FAST_PATH: Contextual harm C_harm={c_harm:.2f} in [{self.C_HARM_CRITICAL_THRESHOLD}, {self.C_HARM_SEVERE_EMERGENCY})")

        # Tier 1 Moderate Physical Risk
        phys_mod_raw = ((self.P10_ALARM_THRESHOLD <= p10 < self.P10_CRITICAL_THRESHOLD) or
                        (self.D10_CRITICAL_THRESHOLD < d10 <= self.D10_MODERATE_THRESHOLD)) and not critical_fast_path

        # Tier 1 Moderate Contextual Warning
        ctx_mod_raw = (self.C_HARM_MODERATE_THRESHOLD <= c_harm < self.C_HARM_CRITICAL_THRESHOLD) and not critical_fast_path

        # Update 2-of-3 rolling filters
        self._physical_risk_history.pop(0)
        self._physical_risk_history.append(phys_mod_raw)
        physical_mod_confirmed = (sum(self._physical_risk_history) >= 2)

        self._contextual_risk_history.pop(0)
        self._contextual_risk_history.append(ctx_mod_raw)
        contextual_mod_confirmed = (sum(self._contextual_risk_history) >= 2)

        if phys_mod_raw:
            if self.P10_ALARM_THRESHOLD <= p10 < self.P10_CRITICAL_THRESHOLD:
                alarms.append(f"MODERATE_RISK: Elevated collision prob P10={p10:.3f}")
            if self.D10_CRITICAL_THRESHOLD < d10 <= self.D10_MODERATE_THRESHOLD:
                alarms.append(f"MODERATE_RISK: Low clearance d10={d10*1000:.1f}mm")
        if ctx_mod_raw:
            alarms.append(f"MODERATE_CONTEXTUAL_WARNING: Contextual harm C_harm={c_harm:.2f}")

        if physical_mod_confirmed:
            alarms.append("MODERATE_PHYSICAL_RISK_CONFIRMED (2-of-3 tactical cycles)")
        if contextual_mod_confirmed:
            alarms.append("MODERATE_CONTEXTUAL_WARNING_CONFIRMED (2-of-3 tactical cycles)")

        return severe_emergency, critical_fast_path, physical_mod_confirmed, contextual_mod_confirmed, alarms

    # -------------------------------------------------------------------------
    # Recovery Preconditions & Degraded Mode
    # -------------------------------------------------------------------------
    def check_recovery_preconditions(
        self,
        current_qvel: np.ndarray,
        planner_stage: str = "MOVE_TO_TARGET"
    ) -> Tuple[bool, bool, List[str]]:
        """
        Verify the 6-point Recovery Preconditions checklist.
        """
        failures = []
        is_degraded = False

        max_vel = np.max(np.abs(current_qvel[:7]))
        if max_vel >= self.REST_VELOCITY_THRESHOLD:
            failures.append(f"Robot not at physical rest (max |qvel|={max_vel:.5f} >= {self.REST_VELOCITY_THRESHOLD} rad/s)")

        if self.latest_reactive is not None:
            valid_r, r_errs = self.latest_reactive.validate()
            if not valid_r:
                failures.append(f"Reactive telemetry invalid: {r_errs}")
            if (self.latest_reactive.has_collision or
                self.latest_reactive.robot_obstacle_collision or
                self.latest_reactive.robot_proxy_collision or
                self.latest_reactive.robot_table_collision):
                failures.append("Layer 0 active collision detected on sensors")
            if self.latest_reactive.geometric_min_distance < self.L0_RECOVERY_DISTANCE_THRESHOLD:
                failures.append(f"Layer 0 clearance ({self.latest_reactive.geometric_min_distance*1000:.1f}mm) < {self.L0_RECOVERY_DISTANCE_THRESHOLD*1000:.0f}mm margin")
        else:
            failures.append("Layer 0 reactive telemetry missing")

        if self._is_latched:
            failures.append("Reactive emergency stop remains latched (operator reset required)")

        if self.latest_tactical is None:
            failures.append("Tactical telemetry missing")
        elif self.latest_tactical.age_ms > self.H10_STALE_MAX_MS:
            failures.append(f"Tactical telemetry expired ({self.latest_tactical.age_ms:.1f}ms > {self.H10_STALE_MAX_MS}ms)")
        else:
            valid_t, t_errs = self.latest_tactical.validate()
            if not valid_t:
                failures.append(f"Tactical telemetry invalid: {t_errs}")
            if self.latest_tactical.p10 >= self.P10_RECOVERY_THRESHOLD:
                failures.append(f"Tactical risk elevated (P10={self.latest_tactical.p10:.2f} >= {self.P10_RECOVERY_THRESHOLD})")
            if self.latest_tactical.d10_clear < self.D50_RECOVERY_THRESHOLD:
                failures.append(f"Tactical clearance low (d10={self.latest_tactical.d10_clear*1000:.1f}mm < {self.D50_RECOVERY_THRESHOLD*1000:.0f}mm)")
            if self.latest_tactical.c_harm >= self.C_HARM_MODERATE_THRESHOLD:
                failures.append(f"Tactical contextual harm elevated (C_harm={self.latest_tactical.c_harm:.2f} >= {self.C_HARM_MODERATE_THRESHOLD})")

        if self.latest_strategic is not None and not self.latest_strategic.is_stale and self.latest_strategic.age_ms <= self.H50_FRESH_MAX_MS:
            valid_s, s_errs = self.latest_strategic.validate()
            if not valid_s:
                failures.append(f"Strategic telemetry invalid: {s_errs}")
            if self.latest_strategic.p50 >= self.P50_RECOVERY_THRESHOLD:
                failures.append(f"Strategic risk elevated (P50={self.latest_strategic.p50:.2f} >= {self.P50_RECOVERY_THRESHOLD})")
            if self.latest_strategic.d50_clear < self.D50_RECOVERY_THRESHOLD:
                failures.append(f"Strategic clearance low (d50={self.latest_strategic.d50_clear*1000:.1f}mm < {self.D50_RECOVERY_THRESHOLD*1000:.0f}mm)")
        else:
            is_degraded = True

        if planner_stage in ["LIFT", "MOVE_TO_TARGET"]:
            if self.latest_reactive is not None and not self.latest_reactive.robot_object_contact:
                failures.append("Payload grasp lost or slipped during stop in transit stage")

        can_recover = (len(failures) == 0)
        return can_recover, is_degraded, failures

    # -------------------------------------------------------------------------
    # Explicit Operator Reset Handshake
    # -------------------------------------------------------------------------
    def clear_latched_emergency_stop(
        self,
        current_qvel: np.ndarray,
        operator_confirm: bool = False
    ) -> Tuple[bool, str]:
        """
        Explicit operator reset handshake to release a latched emergency stop.
        """
        if not operator_confirm:
            return False, "RESET_REJECTED: Operator confirmation flag required"

        if not self._is_latched:
            return True, "RESET_IGNORED: Supervisor is not in a latched state"

        if self.latest_reactive is None:
            return False, "CANNOT_RESET: Reactive telemetry missing. Cannot verify collision clearance."

        valid_r, r_errs = self.latest_reactive.validate()
        if not valid_r:
            return False, f"CANNOT_RESET: Reactive telemetry invalid ({r_errs})"

        if (self.latest_reactive.has_collision or
            self.latest_reactive.robot_obstacle_collision or
            self.latest_reactive.robot_proxy_collision or
            self.latest_reactive.robot_table_collision):
            return False, "CANNOT_RESET: Physical contact still active on collision sensors (obstacle/proxy/table)"

        if self.latest_reactive.geometric_min_distance < 0.10:
            return False, f"CANNOT_RESET: Clearance ({self.latest_reactive.geometric_min_distance*1000:.1f}mm) < 100mm margin"

        if current_qvel is None:
            return False, "CANNOT_RESET: current_qvel is required to verify physical rest"

        try:
            v_arr = np.asarray(current_qvel, dtype=np.float64)
            if v_arr.ndim != 1 or len(v_arr) < 7:
                return False, f"CANNOT_RESET: current_qvel must be 1D with >= 7 elements, got shape {v_arr.shape}"
            if not np.all(np.isfinite(v_arr[:7])):
                return False, "CANNOT_RESET: Non-finite value (NaN/Inf) in current_qvel"
        except Exception as e:
            return False, f"CANNOT_RESET: Malformed current_qvel: {e}"

        max_v = float(np.max(np.abs(v_arr[:7])))
        if max_v >= self.REST_VELOCITY_THRESHOLD:
            return False, f"CANNOT_RESET: Robot joints still have non-zero velocity in simulation ({max_v:.5f} >= {self.REST_VELOCITY_THRESHOLD} rad/s)"

        self._is_latched = False
        self._is_arresting = False
        self._arrest_settle_counter = 0
        self._state = SupervisorState.RECOVERY
        self._recovery_dwell_counter = 0
        self._has_completed_operator_recovery = True
        self._active_alarms.append("OPERATOR_RESET: Latch released. Entering RECOVERY.")
        return True, "RESET_SUCCESSFUL: Latch released. Entering RECOVERY."

    # -------------------------------------------------------------------------
    # State Machine Step & Command Arbitration
    # -------------------------------------------------------------------------
    def step(
        self,
        current_qpos: np.ndarray,
        current_qvel: np.ndarray,
        nominal_q_target: np.ndarray,
        nominal_gripper_target: float,
        reactive: ReactiveTelemetry,
        tactical: Optional[TacticalTelemetry] = None,
        strategic: Optional[StrategicTelemetry] = None,
        planner_stage: str = "MOVE_TO_TARGET",
        gravity_torques: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> InterventionCommand:
        """
        Execute one 100 Hz supervisor evaluation tick.
        """
        self.latest_reactive = reactive
        self.latest_tactical = tactical
        self.latest_strategic = strategic
        self._active_alarms = []
        actuator_inhibit = False

        # ---------------------------------------------------------------------
        # 0. Input Validation (Robot State, Targets, and Telemetry)
        # ---------------------------------------------------------------------
        input_errors = []

        # Validate dt
        valid_dt = (isinstance(dt, (int, float, np.floating)) and np.isfinite(dt) and dt > 0.0)
        if not valid_dt:
            input_errors.append(f"Invalid timestep dt: {dt}")

        # Validate current_qpos
        q_curr = None
        try:
            q_curr_arr = np.asarray(current_qpos, dtype=np.float64)
            if q_curr_arr.ndim == 1 and len(q_curr_arr) >= 7 and np.all(np.isfinite(q_curr_arr[:7])):
                q_curr = q_curr_arr[:7].copy()
            else:
                input_errors.append(f"current_qpos invalid or non-finite: shape={q_curr_arr.shape}")
        except Exception as e:
            input_errors.append(f"Malformed current_qpos: {e}")

        # Validate current_qvel
        q_vel = None
        try:
            q_vel_arr = np.asarray(current_qvel, dtype=np.float64)
            if q_vel_arr.ndim == 1 and len(q_vel_arr) >= 7 and np.all(np.isfinite(q_vel_arr[:7])):
                q_vel = q_vel_arr[:7].copy()
            else:
                input_errors.append(f"current_qvel invalid or non-finite: shape={q_vel_arr.shape}")
        except Exception as e:
            input_errors.append(f"Malformed current_qvel: {e}")

        # Validate nominal_q_target
        q_nom = None
        try:
            q_nom_arr = np.asarray(nominal_q_target, dtype=np.float64)
            if q_nom_arr.ndim == 1 and len(q_nom_arr) >= 7 and np.all(np.isfinite(q_nom_arr[:7])):
                q_nom = q_nom_arr[:7].copy()
            else:
                input_errors.append(f"nominal_q_target invalid or non-finite: shape={q_nom_arr.shape}")
        except Exception as e:
            input_errors.append(f"Malformed nominal_q_target: {e}")

        # Validate nominal_gripper_target
        valid_gripper = (isinstance(nominal_gripper_target, (int, float, np.floating, np.integer)) and np.isfinite(nominal_gripper_target))
        if not valid_gripper:
            input_errors.append(f"Invalid non-finite nominal_gripper_target: {nominal_gripper_target}")

        # Validate reactive telemetry instance
        valid_reactive = False
        if reactive is None or not isinstance(reactive, ReactiveTelemetry):
            input_errors.append(f"reactive must be an instance of ReactiveTelemetry, got {type(reactive)}")
        else:
            valid_r, r_errs = reactive.validate()
            if valid_r:
                valid_reactive = True
            else:
                input_errors.extend([f"CRITICAL_REACTIVE_INVALID: {e}" for e in r_errs])

        # If any input validation error occurred, execute fail-safe emergency arrest immediately!
        if len(input_errors) > 0:
            self._state = SupervisorState.EMERGENCY_STOP
            self._is_latched = True
            self._has_completed_operator_recovery = False
            self._stop_mode = StopMode.LATCHED_REACTIVE
            self._speed_scale = 0.0
            self._is_arresting = False
            self._recovery_dwell_counter = 0

            for err in input_errors:
                self._active_alarms.append(f"INPUT_VALIDATION_FAILURE: {err}")

            # Determine whether payload grasp should be preserved:
            # Grasp clamping (0.0) is ONLY commanded if an object was actually held!
            # If the gripper was empty, closing it during sensor failure could pinch human fingers or fixture edges.
            if valid_reactive:
                holding_object = reactive.robot_object_contact
            else:
                holding_object = (planner_stage in ["LIFT", "MOVE_TO_TARGET"])

            if holding_object:
                target_gripper = 0.0  # Maintain grasp force to prevent payload drop
            elif valid_gripper:
                target_gripper = float(nominal_gripper_target)  # Preserve current gripper position
            else:
                target_gripper = float(self._last_valid_gripper_target)

            # Determine target_qpos and actuator inhibition per specific feedback failure mode:
            actuator_inhibit = False

            # Case A: Valid qpos, invalid qvel (isolated velocity feedback failure)
            # Emergency position-hold response at the known current position.
            # No controlled-braking or grasp-preservation guarantee.
            if q_curr is not None and q_vel is None:
                if self._frozen_qpos is None:
                    self._frozen_qpos = q_curr.copy()
                cmd_qpos = self._frozen_qpos.copy()
                self._active_alarms.append(
                    "VELOCITY_FEEDBACK_LOST_POSITION_HOLD: Invalid current_qvel. Velocity-informed dynamic braking unavailable; "
                    "commanding instantaneous position-setpoint hold at known q_curr. In high-speed scenarios, actuator torque saturation "
                    "and elevated deceleration may occur. No controlled braking or grasp preservation guarantee is claimed."
                )

            # Case B: Invalid qpos and invalid qvel (simultaneous dual state feedback failure)
            # Hold the previous valid command if one exists; otherwise inhibit actuators and latch the emergency state.
            elif q_curr is None and q_vel is None:
                if self._last_valid_cmd_qpos is not None:
                    cmd_qpos = self._last_valid_cmd_qpos.copy()
                    self._active_alarms.append(
                        "DUAL_FEEDBACK_LOST_HOLD_LAST: Simultaneous failure of current_qpos and current_qvel. "
                        "Holding last valid commanded setpoint; motion toward home prevented."
                    )
                elif self._frozen_qpos is not None:
                    cmd_qpos = self._frozen_qpos.copy()
                    self._active_alarms.append(
                        "DUAL_FEEDBACK_LOST_HOLD_FROZEN: Simultaneous failure of current_qpos and current_qvel. "
                        "Holding frozen setpoint."
                    )
                else:
                    cmd_qpos = None
                    actuator_inhibit = True
                    self._active_alarms.append(
                        "CRITICAL_DUAL_STATE_FEEDBACK_FAULT: First-cycle simultaneous position and velocity feedback failure. "
                        "No state feedback or prior setpoint history available. Arm actuators inhibited; hardware power cutoff required."
                    )

            # Case C: Invalid qpos, valid qvel (isolated position feedback failure)
            # Hold the previous valid command if one exists; otherwise inhibit actuators and latch the emergency state.
            elif q_curr is None and q_vel is not None:
                if self._last_valid_cmd_qpos is not None:
                    cmd_qpos = self._last_valid_cmd_qpos.copy()
                    self._active_alarms.append(
                        "POSITION_FEEDBACK_LOST_HOLD_LAST: current_qpos invalid. Holding last commanded setpoint; motion toward home prevented."
                    )
                elif self._frozen_qpos is not None:
                    cmd_qpos = self._frozen_qpos.copy()
                    self._active_alarms.append(
                        "POSITION_FEEDBACK_LOST_HOLD_FROZEN: current_qpos invalid. Holding frozen setpoint."
                    )
                else:
                    cmd_qpos = None
                    actuator_inhibit = True
                    self._active_alarms.append(
                        "CRITICAL_POSITION_FEEDBACK_FAULT: First-cycle position feedback failure with no prior setpoint history. "
                        "No trustworthy physical position available; cannot guarantee safe physical hold. Arm actuators inhibited; hardware power cutoff required."
                    )

            # 3. Valid qpos & qvel, but invalid nominal_q_target (planner output failure):
            #    Reject planner target and freeze / arrest arm to rest at current position.
            elif q_nom is None:
                max_v = np.max(np.abs(q_vel[:7]))
                if max_v >= self.REST_VELOCITY_THRESHOLD:
                    self._is_arresting = True
                    self._arrest_settle_counter = 0
                    try:
                        traj_res = self.generate_braking_trajectory(
                            q_curr, q_vel, gravity_torques=gravity_torques, dt=dt if valid_dt else 0.01
                        )
                        self._braking_trajectory = traj_res.positions if traj_res.positions is not None else []
                        self._braking_step_idx = 0
                        self._braking_success = traj_res.success
                    except Exception as e:
                        self._braking_trajectory = []
                        self._braking_step_idx = 0
                        self._braking_success = False

                    if len(self._braking_trajectory) > 0:
                        cmd_qpos = self._braking_trajectory[0].copy()
                    else:
                        cmd_qpos = q_curr.copy()
                else:
                    cmd_qpos = q_curr.copy()
                    self._frozen_qpos = q_curr.copy()
                self._active_alarms.append(
                    "INVALID_PLANNER_TARGET: nominal_q_target invalid. Arresting / holding at current position."
                )

            # 4. Valid qpos, qvel, target, but invalid reactive telemetry:
            #    Layer 0 proximity / contact feedback lost. Safe emergency arrest at current position.
            else:
                max_v = np.max(np.abs(q_vel[:7]))
                if max_v >= self.REST_VELOCITY_THRESHOLD:
                    self._is_arresting = True
                    self._arrest_settle_counter = 0
                    try:
                        traj_res = self.generate_braking_trajectory(
                            q_curr, q_vel, gravity_torques=gravity_torques, dt=dt if valid_dt else 0.01
                        )
                        self._braking_trajectory = traj_res.positions if traj_res.positions is not None else []
                        self._braking_step_idx = 0
                        self._braking_success = traj_res.success
                    except Exception as e:
                        self._braking_trajectory = []
                        self._braking_step_idx = 0
                        self._braking_success = False

                    if len(self._braking_trajectory) > 0:
                        cmd_qpos = self._braking_trajectory[0].copy()
                    else:
                        cmd_qpos = q_curr.copy()
                else:
                    cmd_qpos = q_curr.copy()
                    self._frozen_qpos = q_curr.copy()

            # Strictly clamp safe target_qpos to hardware limits if a target exists
            if cmd_qpos is not None:
                for i in range(7):
                    cmd_qpos[i] = np.clip(
                        cmd_qpos[i],
                        self.JOINT_LIMITS_LOWER[i] + self.JOINT_LIMIT_SAFETY_MARGIN_RAD,
                        self.JOINT_LIMITS_UPPER[i] - self.JOINT_LIMIT_SAFETY_MARGIN_RAD
                    )
                self._frozen_qpos = cmd_qpos.copy()
                self._last_valid_cmd_qpos = cmd_qpos.copy()
            self._last_valid_gripper_target = target_gripper

            return InterventionCommand(
                state=self._state,
                speed_scale=0.0,
                target_qpos=cmd_qpos,
                target_gripper=target_gripper,
                replan_flag=True,
                is_latched=True,
                stop_mode=StopMode.LATCHED_REACTIVE,
                active_alarms=self.active_alarms,
                actuator_inhibit=actuator_inhibit,
                supervisor=self,
            )

        max_vel = np.max(np.abs(q_vel[:7]))

        # Evaluate tactical alarms
        severe_emergency, crit_fast_path, phys_mod_confirmed, ctx_mod_confirmed, tactical_alarms = self.evaluate_tactical_alarms(tactical)
        self._active_alarms.extend(tactical_alarms)

        # ---------------------------------------------------------------------
        # 1. Layer 0 Reactive Evaluation (Priority 1: Highest)
        # ---------------------------------------------------------------------
        l0_breach = (reactive.has_collision or
                     reactive.robot_obstacle_collision or
                     reactive.robot_proxy_collision or
                     reactive.robot_table_collision or
                     (reactive.geometric_min_distance <= self.L0_CONTACT_DISTANCE_THRESHOLD))

        if l0_breach:
            if not self._is_latched:
                self._is_latched = True
                self._has_completed_operator_recovery = False
                self._frozen_qpos = q_curr.copy()
                self._active_alarms.insert(0, "LAYER_0_BREACH: Physical collision or distance violation! LATCHED EMERGENCY STOP.")
                if max_vel >= self.REST_VELOCITY_THRESHOLD:
                    self._is_arresting = True
                    self._arrest_settle_counter = 0
                    try:
                        traj_res = self.generate_braking_trajectory(
                            q_curr, q_vel, gravity_torques=gravity_torques, dt=dt
                        )
                        self._braking_trajectory = traj_res.positions if traj_res.positions is not None else []
                        self._braking_step_idx = 0
                        self._braking_success = traj_res.success
                        if not traj_res.success:
                            self._active_alarms.append(f"DYNAMIC_ARREST_GENERATION_FAILED: {traj_res.failure_reason}")
                    except Exception as e:
                        self._braking_trajectory = []
                        self._braking_step_idx = 0
                        self._braking_success = False
                        self._active_alarms.append(f"DYNAMIC_ARREST_GENERATION_FAILED: {e}")

                    if len(self._braking_trajectory) == 0:
                        self._braking_trajectory = [self._frozen_qpos.copy()]
                else:
                    self._is_arresting = False
                    self._braking_trajectory = [self._frozen_qpos.copy()]
                    self._braking_step_idx = 0
                    self._braking_success = True

            self._state = SupervisorState.EMERGENCY_STOP
            self._stop_mode = StopMode.LATCHED_REACTIVE

        # ---------------------------------------------------------------------
        # 2. State Machine Transitions (when not in latched emergency stop)
        # ---------------------------------------------------------------------
        if not self._is_latched:

            # Severe Contextual Emergency -> Unlatched EMERGENCY_STOP
            if severe_emergency:
                self._recovery_dwell_counter = 0
                if self._state != SupervisorState.EMERGENCY_STOP:
                    self._state = SupervisorState.EMERGENCY_STOP
                    self._stop_mode = StopMode.PREDICTIVE_EMERGENCY
                    self._active_alarms.append("TRANSITION: -> EMERGENCY_STOP (Severe Contextual Harm / Human Proximity)")
                    if max_vel >= self.REST_VELOCITY_THRESHOLD:
                        self._is_arresting = True
                        self._arrest_settle_counter = 0
                        try:
                            traj_res = self.generate_braking_trajectory(
                                q_curr, q_vel, gravity_torques=gravity_torques, dt=dt
                            )
                            self._braking_trajectory = traj_res.positions if traj_res.positions is not None else []
                            self._braking_step_idx = 0
                            self._braking_success = traj_res.success
                            if not traj_res.success:
                                self._active_alarms.append(f"DYNAMIC_ARREST_GENERATION_FAILED: {traj_res.failure_reason}")
                        except Exception as e:
                            self._braking_trajectory = []
                            self._braking_step_idx = 0
                            self._braking_success = False
                            self._active_alarms.append(f"DYNAMIC_ARREST_GENERATION_FAILED: {e}")

                        if len(self._braking_trajectory) == 0:
                            self._braking_trajectory = [q_curr.copy()]
                    else:
                        self._is_arresting = False
                        self._frozen_qpos = q_curr.copy()

            elif self._state == SupervisorState.EMERGENCY_STOP:
                # Severe contextual hazard has receded while in unlatched EMERGENCY_STOP
                can_recover, is_deg, _ = self.check_recovery_preconditions(q_vel, planner_stage=planner_stage)
                hazard_reappeared = (severe_emergency or crit_fast_path or phys_mod_confirmed or ctx_mod_confirmed)
                required_dwell = 30 if is_deg else 20
                if can_recover and not hazard_reappeared:
                    self._recovery_dwell_counter += 1
                    if self._recovery_dwell_counter >= required_dwell:
                        self._state = SupervisorState.RECOVERY
                        self._stop_mode = StopMode.NONE
                        self._recovery_dwell_counter = 0
                        self._active_alarms.append(f"TRANSITION: EMERGENCY_STOP (Unlatched) -> RECOVERY (Hazard Receded & Preconditions Met {required_dwell} Ticks)")
                else:
                    self._recovery_dwell_counter = 0

            elif self._state == SupervisorState.NORMAL:
                # Missing H10 telemetry -> Fail-Safe Policy: Never assume safe!
                if tactical is None:
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: NORMAL -> SLOWDOWN (Missing H10 Telemetry - Fail-Safe)")
                elif crit_fast_path:
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: NORMAL -> SLOWDOWN (Critical Fast-Path Trigger)")
                elif phys_mod_confirmed:
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: NORMAL -> SLOWDOWN (Moderate Physical Risk Confirmed)")
                elif tactical.age_ms > self.H10_STALE_MAX_MS:
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: NORMAL -> SLOWDOWN (Tactical Telemetry Expired > 80ms)")
                # Caution triggers: contextual warning, strategic conflict, or stale telemetry
                elif (
                    ctx_mod_confirmed
                    or (self.D10_MODERATE_THRESHOLD < tactical.d10_clear <= self.D10_CAUTION_THRESHOLD)
                    or (strategic is not None and strategic.p50 >= self.P50_STRATEGIC_THRESHOLD)
                    or (self.H10_FRESH_MAX_MS < tactical.age_ms <= self.H10_STALE_MAX_MS)
                    or (strategic is None)
                    or (strategic is not None and (strategic.is_stale or strategic.age_ms > self.H50_FRESH_MAX_MS))
                ):
                    self._state = SupervisorState.CAUTION
                    self._stop_mode = StopMode.NONE
                    self._active_alarms.append("TRANSITION: NORMAL -> CAUTION")

            elif self._state == SupervisorState.CAUTION:
                if (tactical is None
                    or crit_fast_path
                    or phys_mod_confirmed
                    or tactical.age_ms > self.H10_STALE_MAX_MS):
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: CAUTION -> SLOWDOWN")
                # De-escalation to NORMAL requires BOTH fresh H10 and fresh H50
                elif (
                    tactical is not None
                    and tactical.d10_clear >= self.D10_RECOVERY_CLEARANCE
                    and tactical.p10 <= self.P10_RECOVERY_THRESHOLD
                    and tactical.c_harm < self.C_HARM_MODERATE_THRESHOLD
                    and tactical.age_ms <= self.H10_FRESH_MAX_MS
                    and strategic is not None
                    and not strategic.is_stale
                    and strategic.age_ms <= self.H50_FRESH_MAX_MS
                    and strategic.p50 <= self.P50_RECOVERY_THRESHOLD
                    and strategic.d50_clear >= self.D50_RECOVERY_THRESHOLD
                ):
                    self._clear_path_dwell_counter += 1
                    if self._clear_path_dwell_counter >= 10:
                        self._state = SupervisorState.NORMAL
                        self._clear_path_dwell_counter = 0
                        self._active_alarms.append("TRANSITION: CAUTION -> NORMAL (Clearance & Fresh H50 Sustained 10 Ticks)")
                else:
                    self._clear_path_dwell_counter = 0

            elif self._state == SupervisorState.SLOWDOWN:
                can_recover, is_deg, _ = self.check_recovery_preconditions(q_vel, planner_stage=planner_stage)
                hazard_reappeared = (severe_emergency or crit_fast_path or phys_mod_confirmed or ctx_mod_confirmed)
                required_dwell = 30 if is_deg else 20
                if can_recover and not hazard_reappeared:
                    self._recovery_dwell_counter += 1
                    if self._recovery_dwell_counter >= required_dwell:
                        self._state = SupervisorState.RECOVERY
                        self._recovery_dwell_counter = 0
                        self._active_alarms.append(f"TRANSITION: SLOWDOWN -> RECOVERY (Preconditions Met {required_dwell} Ticks)")
                else:
                    self._recovery_dwell_counter = 0

            elif self._state == SupervisorState.RECOVERY:
                if crit_fast_path or phys_mod_confirmed or (tactical is None):
                    self._state = SupervisorState.SLOWDOWN
                    self._stop_mode = StopMode.CONTROLLED_DECEL
                    self._recovery_dwell_counter = 0
                    self._active_alarms.append("TRANSITION: RECOVERY -> SLOWDOWN (Hazard Re-appeared or H10 Lost)")
                elif self._speed_scale >= 0.99:
                    can_recover, is_deg, _ = self.check_recovery_preconditions(q_vel, planner_stage=planner_stage)
                    if can_recover and not is_deg:
                        self._state = SupervisorState.NORMAL
                        self._stop_mode = StopMode.NONE
                        self._speed_scale = 1.0
                        self._active_alarms.append("TRANSITION: RECOVERY -> NORMAL (Speed Ramp Completed & Fresh H50 Verified)")

        # ---------------------------------------------------------------------
        # 3. Dynamic Arrest and Command Calculation per State
        # ---------------------------------------------------------------------
        replan_flag = False

        # Reset settle counter if velocity exceeds threshold
        if max_vel >= self.REST_VELOCITY_THRESHOLD:
            self._arrest_settle_counter = 0

        # Gripper safety protection:
        # In EMERGENCY_STOP or SLOWDOWN, clamp gripper (target=0.0) ONLY if holding an object
        # to prevent dropping the payload. If empty, preserve current target to prevent pinching.
        target_gripper = float(nominal_gripper_target)
        if self._state in [SupervisorState.EMERGENCY_STOP, SupervisorState.SLOWDOWN]:
            if reactive.robot_object_contact:
                target_gripper = 0.0

        if self._state == SupervisorState.NORMAL:
            self._speed_scale = 1.0
            cmd_qpos = q_nom.copy()
            self._stop_mode = StopMode.NONE

        elif self._state == SupervisorState.CAUTION:
            if tactical is not None and self.H10_FRESH_MAX_MS < tactical.age_ms <= self.H10_STALE_MAX_MS:
                self._speed_scale = 0.50
            elif strategic is None or strategic.is_stale or strategic.age_ms > self.H50_FRESH_MAX_MS:
                self._speed_scale = 0.80
            else:
                self._speed_scale = 0.60

            cmd_qpos = q_curr + self._speed_scale * (q_nom - q_curr)
            if strategic is not None and strategic.p50 >= self.P50_STRATEGIC_THRESHOLD:
                replan_flag = True

        elif self._state == SupervisorState.SLOWDOWN:
            alpha_decel = 2.0
            self._speed_scale = max(0.0, self._speed_scale - alpha_decel * dt)

            if self._speed_scale > 0.0:
                cmd_qpos = q_curr + self._speed_scale * (q_nom - q_curr)
            else:
                if self._frozen_qpos is None:
                    self._frozen_qpos = q_curr.copy()
                cmd_qpos = self._frozen_qpos.copy()

            replan_flag = True

        elif self._state == SupervisorState.EMERGENCY_STOP:
            self._speed_scale = 0.0
            replan_flag = True

            # Determine safe fallback position
            if self._frozen_qpos is not None:
                fallback_hold = self._frozen_qpos.copy()
            elif q_curr is not None:
                fallback_hold = q_curr.copy()
            elif self._last_valid_cmd_qpos is not None:
                fallback_hold = self._last_valid_cmd_qpos.copy()
            else:
                fallback_hold = np.zeros(7, dtype=np.float64)

            if self._is_arresting:
                if len(self._braking_trajectory) > 0 and self._braking_step_idx < len(self._braking_trajectory):
                    cmd_qpos = self._braking_trajectory[self._braking_step_idx].copy()
                    self._braking_step_idx += 1
                elif len(self._braking_trajectory) > 0:
                    # Trajectory exists and has reached its end (exhausted)
                    cmd_qpos = self._braking_trajectory[-1].copy()
                    if max_vel >= self.REST_VELOCITY_THRESHOLD:
                        self._active_alarms.append(
                            f"DYNAMIC_ARREST_EXHAUSTED_STILL_MOVING: Braking trajectory exhausted at step {self._braking_step_idx} "
                            f"but robot velocity (max |qvel|={max_vel:.5f} rad/s) >= {self.REST_VELOCITY_THRESHOLD} rad/s"
                        )
                else:
                    # Trajectory is completely empty (failed generation)
                    cmd_qpos = fallback_hold.copy()
                    self._active_alarms.append(
                        "DYNAMIC_ARREST_EMPTY_TRAJECTORY: No braking trajectory available. Holding fallback position."
                    )

                if max_vel < self.REST_VELOCITY_THRESHOLD:
                    self._arrest_settle_counter += 1
                    if self._arrest_settle_counter >= 10:
                        self._is_arresting = False
                        if len(self._braking_trajectory) > 0:
                            self._frozen_qpos = self._braking_trajectory[-1].copy()
                        else:
                            self._frozen_qpos = fallback_hold.copy()
                        cmd_qpos = self._frozen_qpos.copy()
                        self._active_alarms.append("DYNAMIC_ARREST_SETTLED: Physical rest confirmed for 10 consecutive ticks")
                else:
                    self._arrest_settle_counter = 0
            else:
                if self._frozen_qpos is None:
                    self._frozen_qpos = fallback_hold.copy()
                cmd_qpos = self._frozen_qpos.copy()

        elif self._state == SupervisorState.RECOVERY:
            _, is_deg, _ = self.check_recovery_preconditions(q_vel, planner_stage=planner_stage)
            max_recovery_speed = 0.50 if is_deg else 1.00

            # Clamp immediately to max_recovery_speed if currently above it
            if self._speed_scale > max_recovery_speed:
                self._speed_scale = max_recovery_speed
            else:
                self._speed_scale = min(max_recovery_speed, self._speed_scale + 1.0 * dt)

            cmd_qpos = q_curr + self._speed_scale * (q_nom - q_curr)

            # Detours remain disabled without fresh, valid H50
            if (strategic is not None
                and not strategic.is_stale
                and strategic.age_ms <= self.H50_FRESH_MAX_MS):
                valid_s, _ = strategic.validate()
                if valid_s and strategic.p50 >= self.P50_STRATEGIC_THRESHOLD:
                    replan_flag = True
                else:
                    replan_flag = False
            else:
                replan_flag = False

        # Strict joint limit safety clamping
        if cmd_qpos is not None:
            for i in range(7):
                cmd_qpos[i] = np.clip(
                    cmd_qpos[i],
                    self.JOINT_LIMITS_LOWER[i] + self.JOINT_LIMIT_SAFETY_MARGIN_RAD,
                    self.JOINT_LIMITS_UPPER[i] - self.JOINT_LIMIT_SAFETY_MARGIN_RAD
                )
            self._last_valid_cmd_qpos = cmd_qpos.copy()
        self._last_valid_gripper_target = target_gripper

        return InterventionCommand(
            state=self._state,
            speed_scale=self._speed_scale,
            target_qpos=cmd_qpos,
            target_gripper=target_gripper,
            replan_flag=replan_flag,
            is_latched=self._is_latched,
            stop_mode=self._stop_mode,
            active_alarms=self.active_alarms,
            actuator_inhibit=actuator_inhibit,
            supervisor=self,
        )
