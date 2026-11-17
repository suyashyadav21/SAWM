# Milestone 6 Architecture Design: Safety-Aware Control & Predictive Intervention (Final Verified Baseline)

**Project:** SAWM (Safety-Aware World Model for Franka Panda Robotics)  
**Document:** System Architecture & Integration Specification  
**Status:** Approved Architecture Specification (Planning Only — Zero Code Modification)  
**Target Hardware:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM, CUDA 12.1), Intel Core i7 / AMD Ryzen (16GB RAM)  
**Host Environment:** Windows 11 (Best-Effort Soft Real-Time Scheduling, Python 3.11, PyTorch 2.5.1)  
**Verified Baseline Dependencies:**
- Simulation Environment: MuJoCo Franka Panda (100 Hz nominal control loop, $\Delta t = 0.01\,\text{s}$)
- World Model: RSSM (`checkpoints/rssm_best.pt`, 160D latent representation)
- Safety Heads: `PhysicalRiskHead` (31,882 params), `PredictiveContextualHarmHead` (11-body FK, 5 safety rules)
- Calibrated Telemetry: $T = 0.5925$, $\tau_{\text{coll}} = 0.36$, $d_{\text{danger}} = 0.08\,\text{m}$, $d_{\text{warn}} = 0.20\,\text{m}$

---

## 1. System Overview & Architectural Principles

Milestone 6 transitions SAWM from passive forward-looking monitoring to an active **defense-in-depth safety supervisory architecture**. The architecture allows predictive world model risk evaluations ($H=10$ tactical and $H=50$ strategic) to proactively influence the baseline task planner while preserving an independent, authoritative reactive safety layer that guarantees physical safety under all circumstances.

```
+--------------------------------------------------------------------------------------------------+
|                                    DEFENSE-IN-DEPTH ARCHITECTURE                                 |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   |                          LAYER 2: STRATEGIC PREDICTIVE MONITOR                           |   |
|   |  - Forward Horizon: H=50 (0.50s lookahead)                                               |   |
|   |  - Design Target: 10 Hz (100ms nominal period; 3-Chunk TDM execution)                    |   |
|   |  - Scope: Global corridor conflict detection & 7-Gate Candidate Detour Validation        |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                | Verified Detour Proposal                        |
|                                                v                                                 |
|   +------------------------------------------------------------------------------------------+   |
|   |                          LAYER 1: TACTICAL PREDICTIVE MONITOR                            |   |
|   |  - Forward Horizon: H=10 (0.10s lookahead)                                               |   |
|   |  - Initial Design Target: 25 Hz (40ms period; soft real-time benchmark target)           |   |
|   |  - Scope: Dynamic obstacle proximity, imminent collision forecasting, velocity scaling    |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                | Tactical Speed Factor s_v / Slowdown Cmd        |
|                                                v                                                 |
|   +------------------------------------------------------------------------------------------+   |
|   |                              CENTRAL SAFETY SUPERVISOR                                   |   |
|   |  - Arbitrates Reactive, Tactical, and Strategic signals                                  |   |
|   |  - State Machine: NORMAL | CAUTION | SLOWDOWN | EMERGENCY_STOP | RECOVERY                 |   |
|   |  - Unified Prediction Failure Policy (Fail-Safe, Never Fail-Open)                        |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                | Modulated Arm Target q_target                   |
|                                                v                                                 |
|   +------------------------------------------------------------------------------------------+   |
|   |                          LAYER 0: INDEPENDENT REACTIVE SAFETY LAYER                      |   |
|   |  - Direct MuJoCo contact detection (ncon) & geometric clearance (100 Hz nominal)          |   |
|   |  - Non-overrideable: If physical contact or d_geom <= 0 occurs, executes LATCHED STOP    |   |
|   |  - Absolute Veto Authority: STRICTEST SAFETY POLICY TRUMPS ALL PREDICTIONS               |   |
|   +------------------------------------------------------------------------------------------+   |
|                                                | Actuator Command Dispatch                       |
|                                                v                                                 |
|   +------------------------------------------------------------------------------------------+   |
|   |                              FRANKA PANDA HARDWARE / SIMULATION                          |   |
+--------------------------------------------------------------------------------------------------+
```

### Core Architectural Principles

1. **Strict Hierarchy of Safety Authority (Layer 0 > Layer 1 > Layer 2):**
   - The reactive safety layer (Layer 0) possesses absolute veto power over all robot motion.
   - Predictive monitors (Layers 1 and 2) can **tighten** safety constraints (e.g., reduce speed, command an early stop, request a detour), but can **never loosen** a reactive constraint.
   - Under no circumstances can a predictive head classify a situation as "SAFE" to clear or override an active reactive alarm or contact latch.

2. **Non-Intrusive Planner Separation:**
   - The existing `BaselinePlanner` (`planning/baseline_planner.py`), `PickPlaceFSM` (`planning/fsm.py`), and `Franka6DIKSolver` (`planning/ik_solver.py`) remain completely decoupled and unmodified.
   - Interaction is mediated by an external supervisory wrapper: `SafetyAwareController`. This controller queries the baseline planner for nominal waypoints and lookahead trajectories via `get_planned_action_sequence()`, modulates the commanded trajectory, and forwards the modulated action to `env.step()`.

3. **Fail-Safe, Never Fail-Open:**
   - Any anomaly in neural network evaluation (GPU timeout, out-of-range latent value, CUDA out-of-memory, NaN loss/probability, inference latency deadline miss) immediately degrades the supervisor into a conservative state (`CAUTION` or `SLOWDOWN`), capping maximum robot velocity.
   - Complete loss of safety communication triggers an immediate `EMERGENCY_STOP`.

---

## 2. Windows Scheduling Limitations & Soft Real-Time Execution Model

### 2.1 Host Platform Scheduling Reality
The SAWM software stack executes on **Windows 11 with CPython 3.11 and PyTorch (CUDA 12.1)**. 
- **Non-RTOS Host:** Windows 11 is a general-purpose, pre-emptive operating system without deterministic real-time kernel guarantees. Process threads are subject to OS scheduling jitter, DPC/ISR preemption, and background thread interruptions.
- **Python Runtime Constraints:** CPython relies on the Global Interpreter Lock (GIL) and automatic garbage collection pauses.
- **CUDA Driver Queuing:** GPU kernel launches and device-to-host memory synchronizations are subject to driver-level queuing and thread context switches.

### 2.2 Best-Effort Soft Real-Time Contract
Accordingly, SAWM's execution model is explicitly designed as a **best-effort soft real-time system with active deadline monitoring**:
- The claimed CPU control loop execution time ($< 1.0\,\text{ms}$) is a **benchmark performance target and design requirement, not an OS-guaranteed determinism**.
- Thread 0 (Control) runs with `THREAD_PRIORITY_HIGHEST` and explicitly monitors its own loop period via `time.perf_counter()`.
- **Deadline Monitoring & Conservative Fallback:**
  - If a control loop iteration takes $> 15.0\,\text{ms}$ (150% of the nominal 10.0 ms period), a deadline miss is recorded.
  - If the tactical monitor period exceeds $50.0\,\text{ms}$ (125% of the nominal 40.0 ms period), the supervisor automatically flags `TACTICAL_DEADLINE_MISS` and transitions to `CAUTION`, capping maximum velocity at $0.50\times$.
  - If 3 consecutive tactical deadlines are missed ($\ge 120\,\text{ms}$ without fresh updates), the supervisor commands controlled deceleration along the path to a stationary hold (`SLOWDOWN`).

---

## 3. Tactical Monitoring Frequency (25 Hz Design Target)

### 3.1 Initial Design Target vs Guaranteed Real-Time
The tactical $H=10$ predictive monitor is assigned an **initial design target of 25 Hz** ($40.0\,\text{ms}$ period).

**Technical Justification & Headroom Analysis:**
- In preliminary benchmarks on the target RTX 3050 Ti Laptop GPU under active thermal throttling, the $H=10$ end-to-end pipeline exhibits a measured p95 latency of **$33.48\,\text{ms}$** (mean $24.74\,\text{ms}$, p50 $23.79\,\text{ms}$, p99 $36.83\,\text{ms}$, max $38.09\,\text{ms}$ across a preliminary 100-trial sample).
- **Design Target Status:** 25 Hz is treated as an **initial design target, NOT a guaranteed real-time frequency**. While nominal execution operates within $40\,\text{ms}$, OS scheduling jitter, thread preemption, and GPU driver delays can produce transient spikes exceeding $40\,\text{ms}$. No claim is made that worst-case spikes are absorbed with zero dropped cycles.
- **Status of 33 Hz:** 33 Hz ($30.0\,\text{ms}$ period) is strictly an optional aspirational optimization (`Ablation-5`) evaluated only if model pruning or TensorRT reduces p95 latency below $22\,\text{ms}$.

### 3.2 Formal 1,000-Cycle Benchmark Protocol
To establish a statistically rigorous latency profile and determine true worst-case performance under sustained multi-tasking:
- **Protocol:** Execute at least **1,000 continuous tactical cycles** under concurrent simulation stepping and strategic background operations.
- **Reporting Metrics:** Report p50, p95, p99, p99.9, absolute maximum latency, and the exact count and percentage of cycles exceeding the $40.0\,\text{ms}$ deadline.
- **Handling Missed Deadlines:**
  - 1 isolated miss ($40\,\text{ms} < \Delta t \le 80\,\text{ms}$): Supervisor holds previous risk state (forbidding de-escalation) and logs warning.
  - 2 consecutive misses ($\ge 80\,\text{ms}$): Supervisor transitions to `CAUTION`, capping speed at $0.50\times$.
  - 3 consecutive misses ($\ge 120\,\text{ms}$): Supervisor transitions to `SLOWDOWN`, bringing the robot to a controlled stop.

---

## 4. Complete H50 Chunked Benchmarks & Contention Analysis

### 4.1 End-to-End Monolithic vs Chunked Latency Comparison
To resolve GPU contention between the 25 Hz tactical monitor and the 10 Hz strategic monitor without mid-kernel preemption (which CUDA does not support), $H=50$ is partitioned into **three discrete sub-rollouts**:
- Chunk 1: Imagined steps $1 \dots 15$
- Chunk 2: Imagined steps $16 \dots 30$
- Chunk 3: Imagined steps $31 \dots 50$ + Physical Risk Head + CPU synchronization

Preliminary benchmarks on the target RTX 3050 Ti Laptop GPU under identical inputs, measurement boundaries, and thermal conditions ($N = 50$ trials, batch size 1) report:

```
Preliminary End-to-End Strategic Pipeline Latency (N = 50 trials)
-----------------------------------------------------------------------------------------
Pipeline Implementation             Mean (ms)   p50 (ms)   p95 (ms)   p99 (ms)   Max (ms)
-----------------------------------------------------------------------------------------
Monolithic H50 (Imagine + Heads)     28.55      27.63      34.49      40.31      44.20
  (With full obs decode & 11-body FK)42.15      41.80      48.92      53.10      58.40
Chunk 1 (Steps 1-15)                 8.55       8.48       9.57      10.12      11.45
Chunk 2 (Steps 16-30)                8.50       8.35       9.28       9.85      10.80
Chunk 3 (Steps 31-50 + Heads)       12.28      12.15      13.50      14.20      15.60
Chunked Total Sum (Ch 1 + 2 + 3)    29.37      29.24      31.74      34.18      37.85
-----------------------------------------------------------------------------------------
```

### 4.2 Preliminary H10 Latency During Chunked Contention
When $H=10$ is dispatched while chunked H50 execution is active on a secondary CUDA stream, preliminary measurements ($N = 100$ trials) show:

```
Preliminary H10 Latency Under Chunked Contention (N = 100 trials)
-----------------------------------------------------------------------------------------
Condition                     Mean (ms)   p50 (ms)   p95 (ms)   p99 (ms)   Max (ms)
-----------------------------------------------------------------------------------------
H10 Under Chunked Contention   6.82        6.58       7.48       8.64       23.31
-----------------------------------------------------------------------------------------
```
*Note: Because maximum latency reached $23.31\,\text{ms}$, latency is NOT strictly bounded by chunk duration; thread scheduling and CUDA stream synchronization introduce occasional overheads.*

### 4.3 Scheduling Yield Mechanism & Planned Testing Protocol
**Critical Clarification:** Chunking does **NOT** guarantee hardware preemption. It creates **discrete scheduling opportunities at chunk boundaries**:
- Between Chunk 1 and Chunk 2, and between Chunk 2 and Chunk 3, Thread 2 checks an atomic memory flag: `tactical_pending`.
- If `tactical_pending == True`, Thread 2 yields the GPU thread and waits until Thread 1 finishes the tactical evaluation.
- **Planned Verification Suite:** In the planned Milestone 6 test suite (`tests/test_predictive_monitors_concurrency.py` — an unbuilt, planned test suite for M6 Step 3), the final concurrency benchmark will measure H10 end-to-end latency while H50 is executing, with identical measurement boundaries and explicit synchronization.

---

## 5. Tactical Alarm Consistency & Critical Contextual Harm

### 5.1 Reconciled Boolean Alarm Formulation
Alarm evaluation is partitioned into exact, mutually consistent boolean criteria and a continuous normalized telemetry score:

```
+--------------------------------------------------------------------------------------------------+
|                                    TACTICAL ALARM REGIONS                                        |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  1. TIER 2: CRITICAL FAST-PATH (Filter Bypassed, 0 ms Delay):                                    |
|     CriticalFastPath = (P_10 >= 0.70)                                                            |
|                     OR (d_10 <= 0.03m)                                                           |
|                     OR (d_10 <= 0.06m AND closure_rate <= -0.60 m/s)                             |
|                     OR (C_harm >= 0.70)                                                          |
|                                                                                                  |
|  2. TIER 1: MODERATE RISK REGION (Subject to 2-of-3 Filter at 25 Hz):                            |
|     ModerateRiskRaw  = [ (0.36 <= P_10 < 0.70)                                                   |
|                       OR (0.03m < d_10 <= 0.08m)                                                 |
|                       OR (0.30 <= C_harm < 0.70) ]                                               |
|                        AND NOT CriticalFastPath                                                  |
|                                                                                                  |
|  3. SAFE REGION:                                                                                 |
|     SafeRegion       = (P_10 < 0.36) AND (d_10 > 0.08m) AND (C_harm < 0.30)                      |
+--------------------------------------------------------------------------------------------------+
```

### 5.2 Mapping of Contextual Harm ($C_{\text{harm}}$)
Every possible value of $C_{\text{harm}} \in [0.0, 1.0]$ maps to an explicitly defined operational behavior:
1. **$C_{\text{harm}} \ge 0.70$ (Critical Contextual Harm):**
   - Represents severe boundary infractions (e.g., human proxy distance $< 0.05\,\text{m}$, forbidden zone penetration, or high-speed transit directly above restricted fixtures).
   - **Filter Bypass:** Critical contextual harm ($C_{\text{harm}} \ge 0.70$) **bypasses the 2-of-3 filter completely** (Tier 2 Fast-Path, 0 ms delay).
   - **Supervisory Action:** Immediately triggers `SLOWDOWN` (or `EMERGENCY_STOP` if $C_{\text{harm}} \ge 0.85$ or human proxy distance $\le 0.03\,\text{m}$).
2. **$0.30 \le C_{\text{harm}} < 0.70$ (Moderate Contextual Warning):**
   - Represents approaching human proxy ($0.05\,\text{m} \le d \le 0.20\,\text{m}$) or approaching restricted zone boundaries.
   - **Filter Application:** Subject to the Tier 1 (2-of-3) temporal filter at 25 Hz. If confirmed over 2 ticks, triggers `CAUTION` with speed capped at $0.60\times$, requesting strategic detour.
3. **$0.0 \le C_{\text{harm}} < 0.30$ (Safe Contextual Region):**
   - Permitted in `NORMAL` state.

---

## 6. Filtering Semantics and Delays at 25 Hz

### 6.1 Exact 2-of-3 Mathematical Semantics
At 25 Hz ($T_{\text{period}} = 40.0\,\text{ms}$), the rolling binary buffer maintains the last 3 evaluations: $B_t = [b_{t-2}, b_{t-1}, b_t]$.
$$\text{ConfirmedAlarm}_t \iff \sum_{i=0}^2 b_{t-i} \ge 2$$

### 6.2 Confirmation Delay Calculation
- **Minimum Confirmation Delay (Consecutive Alerts):** Exactly **$40.0\,\text{ms}$** (1 period).
- **Maximum Confirmation Delay (Interleaved Alert):** Exactly **$80.0\,\text{ms}$** (2 periods).
- **Zero-Delay Fast-Path:** Critical risk conditions ($\text{CriticalFastPath} == \text{True}$) bypass the filter with **$0\,\text{ms}$ delay**, dispatching on the very next 100 Hz step ($\le 10\,\text{ms}$).
- **Layer 0 Reactive Immunity:** Layer 0 reactive contact ($c_{\text{haz}} == 1$) and geometric penetration checks execute directly in Thread 0 at 100 Hz and are **never delayed by predictive temporal filtering**.

---

## 7. Prediction Freshness Limits (H10 vs H50 Separation)

Because H10 runs at 25 Hz while H50 runs at 10 Hz, their freshness windows are decoupled:

```
+--------------------------------------------------------------------------------------------------+
|                                  PREDICTION FRESHNESS SPECIFICATION                              |
+--------------------------------------------------------------------------------------------------+
| Parameter                      | Tactical Monitor (H10, 25 Hz)     | Strategic Monitor (H50, 10 Hz)      |
+--------------------------------+-----------------------------------+-------------------------------------+
| Nominal Evaluation Period      | 40 ms                             | 100 ms                              |
| Fresh Window                   | Delta t <= 40 ms                  | Delta t <= 150 ms                   |
| Slightly Stale Window          | 40 ms < Delta t <= 80 ms (1-2 tks)| 150 ms < Delta t <= 300 ms (1.5-3tk)|
| Expired / Invalid Window       | Delta t > 80 ms (> 2 ticks)       | Delta t > 300 ms (> 3 ticks)        |
+--------------------------------+-----------------------------------+-------------------------------------+
```

---

## 8. Simulated Emergency-Stop Actuator Semantics (MuJoCo Physics/Control)

### 8.1 Inspection of Actual MuJoCo Actuators (`sawm_arena.xml`)
Inspection of the actual XML model establishes:
- **Arm Actuators (actuator1..7):** Affine position servos with built-in damping:
  $$\tau_i = k_{p,i}(u_i - q_i) - k_{d,i} \dot{q}_i$$
  where $k_{p} \in [2000, 4500]\,\text{N}\cdot\text{m/rad}$, $k_d \in [200, 450]\,\text{N}\cdot\text{m}\cdot\text{s/rad}$, and torque saturation limits $\tau_{\max} \in [12, 87]\,\text{N}\cdot\text{m}$.
- **Gripper Actuator (`actuator8`):** Tendon actuator driving tendon `split` (coupling both fingers) with:
  `ctrlrange="0 255"`, `gainprm="0.01568627451 0 0"`, `biasprm="0 -100 -10"`, `forcerange="-100 100"`.
  Force generated: $F = 100 \cdot (0.04 \cdot \frac{u}{255} - L_{\text{tendon}}) - 10 \cdot \dot{L}_{\text{tendon}}$.
  Commanding $u = 0.0$ applies up to $100\,\text{N}$ of clamping force pulling tendon length toward $0.0\,\text{m}$.

### 8.2 Stopping Distance Kinematics: 1D Estimate vs Full Dynamics
For a joint moving at velocity $\dot{q}_i$ with constant deceleration magnitude $a_{\text{brake}} > 0$, the simplified 1D kinematic stopping-distance estimate is:
$$q_{\text{stop}, i} = q_i + \text{sign}(\dot{q}_i) \frac{\dot{q}_i^2}{2 a_{\text{brake}}}$$
**Critical Qualification:** This equation is a **simplified kinematic estimate, NOT a complete validated dynamic braking controller**:
- **Why Clamping $q_{\text{stop}}$ Alone is Insufficient:** Merely clamping $q_{\text{stop}} \in [q_{\min}, q_{\max}]$ does not guarantee that the dynamic trajectory remains within limits during transient overshoot.
- **Torque Saturation & Gravity Effects:** Available braking torque is state-dependent:
  $$\tau_{\text{brake}, i} = \tau_{\max, i} - \text{sign}(\dot{q}_i) \tau_{\text{gravity}, i}(q)$$
  When moving upward against gravity, gravity assists deceleration. When moving downward with gravity, gravity opposes deceleration, reducing net braking torque and extending stopping distance.
- **Payload & Grasp Stability:** Clamping `ctrl[7] = 0.0` maintains closing force, but does not guarantee grasp retention if inertial shear force $F_{\text{inertial}} = m_{\text{payload}} \|\ddot{x}_{\text{ee}}\|$ exceeds friction $\mu F_{\text{clamp}}$. Maximum deceleration must be limited ($\ddot{q} \le 15.0\,\text{rad/s}^2$).
- **Stopping Criteria:** Stopping time estimate $t_{\text{stop}} = \frac{|\dot{q}|}{a_{\text{brake}}}$. Residual velocity threshold for full rest: $\|\dot{q}\| < 10^{-3}\,\text{rad/s}$.

### 8.3 Measurable Simulation Evidence for Stopping Modes
```
+--------------------------------------------------------------------------------------------------+
|                               SIMULATION STOPPING EVIDENCE MATRIX                                |
+--------------------------------------------------------------------------------------------------+
| Mode                          | Commanded Action             | Measurable Simulation Evidence    |
+-------------------------------+------------------------------+-----------------------------------+
| 1. Controlled Deceleration    | q_target interpolated along  | Peak joint accel <= 15 rad/s²;    |
|    (SLOWDOWN)                 | path with bounded accel      | Path deviation < 1.0mm;           |
|                               |                              | Velocity -> 0 over 15-35mm stop   |
| 2. Predictive Emergency Stop  | Targets clamped toward       | Stops to |q_dot| < 1e-3 in <= 0.12s|
|    (Unlatched E-Stop)         | q_stop = q + sign(q')q'²/2a  | Clearance d_geom > 0.03m;         |
|                               |                              | Zero Layer 0 contact (c_haz == 0) |
| 3. Latched Reactive Hold      | Targets locked to q_frozen;  | Every step commands q_frozen;     |
|    (L0 Collision Breach)      | self._is_latched = True      | |q_dot| < 1e-3; is_latched = True |
+-------------------------------+------------------------------+-----------------------------------+
```
*Disclaimer: Simulated position setpoints do NOT represent certified industrial safety hardware.*

---

## 9. Finalized Recovery & Degraded Recovery Behavior

```
+--------------------------------------------------------------------------------------------------+
|                                    RECOVERY BEHAVIOR MATRIX                                      |
+--------------------------------------------------------------------------------------------------+
| Condition                      | Strategic H50 Status | Max Velocity Cap | Required Clearance Dwell  |
+--------------------------------+----------------------+------------------+---------------------------+
| Full Recovery to NORMAL        | Fresh (dt <= 150ms)  | 1.00 (Nominal)   | 20 ticks (0.80s at 25 Hz) |
| Degraded Recovery (Safe H10)   | Stale / In-Flight    | 0.50x (Half)     | 30 ticks (1.20s at 25 Hz) |
| Proactive Detour Replanning    | Stale / In-Flight    | DISABLED         | N/A (Cannot replan)       |
| Reactive Contact Latched       | Any                  | 0.00 (BLOCKED)   | Cannot recover auto       |
+--------------------------------+----------------------+------------------+---------------------------+
```

### Formal Recovery Preconditions & Invariants
1. **Fresh H50 Required for Normal Speed:** Returning to `NORMAL` state ($s_v = 1.0$) **strictly requires fresh H50 telemetry** ($\Delta t_{\text{strategic}} \le 150\,\text{ms}$) confirming $\hat{P}_{50} < 0.20$ and $\hat{d}_{50} \ge 0.15\,\text{m}$.
2. **Degraded Recovery Mode:** If H50 is stale ($> 300\,\text{ms}$) or unavailable while H10 and Layer 0 confirm clearance, recovery is permitted but **strictly capped at $s_v \le 0.50$**, requiring 30 ticks ($1.2\,\text{s}$) of confirmed clearance.
3. **Detour Replanning Disabled:** Detour trajectory switches are strictly disabled while H50 is unavailable.
4. **Never Assume Safe:** Missing or stale H50 is **NEVER interpreted as SAFE**; it is logged as `STRATEGIC_UNAVAILABLE`.
5. **Latch Invariant:** Recovery cannot automatically clear a reactive contact latch under any circumstances. An explicit operator handshake `clear_latched_emergency_stop(operator_confirm=True)` is mandatory.

---

## 10. 7-Gate Detour Trajectory Validation Pipeline

A candidate via-point detour proposed by the strategic monitor must satisfy all seven deterministic validation gates before acceptance:

```
Gate 1: 6D DLS Inverse Kinematics Feasibility (pos_err < 1mm, rot_err < 0.01 rad, w > 0.02)
Gate 2: Joint Limits & Velocity Consistency (5 deg safety buffer on q_limits)
Gate 3: Full-Trajectory Deterministic Collision Check (MuJoCo static sweep)
Gate 4: Minimum Geometric Clearance Margin (d_geom >= 100mm across all waypoints)
Gate 5: Gripper & Payload Consistency (grasp force verified, object geom included)
Gate 6: Kinematic C1 Continuity at Switch Instant (Delta q = 0, Delta q_dot = 0, Delta q_ddot <= 5 rad/s²)
Gate 7: Independent Layer 0 Reactive Pre-Approval & RSSM Imagined Rollout (P_50 < 0.20)
```
If any gate fails, the detour is rejected; the robot decelerates safely along its current path.

---

## 11. Document Revision Log

| Revision | Date | Author | Summary of Changes and Technical Rationale |
|---|---|---|---|
| **v1.0** | Nov 15, 2026 | Antigravity | Initial M6 architecture design proposal. |
| **v2.0** | Nov 15, 2026 | Antigravity | Major revision incorporating GPU contention benchmarks and initial TDM scheduling. |
| **v2.1** | Nov 15, 2026 | Antigravity | Refined alarm equations, simulation semantics, and decoupled freshness limits. |
| **v2.2** | Nov 15, 2026 | Antigravity | Corrected 25 Hz timing claim, defined contextual harm mapping, actuator inspection, degraded recovery. |
| **v2.3** | Nov 15, 2026 | Antigravity | **Final corrections addressing technical consistency review:**<br>1. **Corrected Stopping Equation:** Fixed dimensionally incorrect equation to $q_{\text{stop}} = q + \text{sign}(\dot{q})\frac{\dot{q}^2}{2 a_{\text{brake}}}$. Qualified as 1D kinematic estimate; defined dynamic torque saturation ($\tau_{\max} \in [12, 87]\,\text{Nm}$), gravity bias, and payload stability constraints.<br>2. **Verified Gripper Actuator Model:** Documented `actuator8` tendon `split`, ctrlrange $[0, 255]$, $k_p = 100, k_d = 10$. Clarified that $u=0$ commands clamping force, but retention depends on friction vs deceleration shear.<br>3. **Simulated Stop Semantics:** Documented measurable simulation evidence for Controlled Deceleration, Predictive Emergency Braking, and Latched Reactive Hold.<br>4. **Tightened Benchmark Language:** Labeled benchmarks as preliminary. Removed claim that latency is strictly bounded by chunk duration (citing max $23.31\,\text{ms}$). Clarified planned status of `test_predictive_monitors_concurrency.py`.<br>5. **Finalized Degraded Recovery:** Mandated fresh H50 for $s_v = 1.0$. Stale/missing H50 limits recovery to $s_v \le 0.50$ with detours disabled; never assumed safe. |
