# Milestone 6 Experiment Plan: Comparative Safety & Intervention Benchmarks (Final Verified Baseline)

**Project:** SAWM (Safety-Aware World Model for Franka Panda Robotics)  
**Document:** Experimental Design & Evaluation Protocol  
**Status:** Approved Architecture Specification (Planning Only — Zero Code Modification)  
**Evaluation Scope:** 4-Way Comparative Study Across 5 Hazard Scenarios  

---

## 1. Objective and Hypothesis

The primary objective of Milestone 6 evaluation is to empirically prove that **predictive safety intervention** (using the world model and trained safety heads) prevents hazardous physical collisions and near-misses that conventional reactive safety either misses or arrests too late, while maintaining high task completion efficiency and avoiding false interventions on safe trajectories.

### Core Hypotheses to Test
1. **Hypothesis 1 (Collision Prevention):** Combined Predictive + Reactive safety achieves a **0.0% collision rate** across dynamic and static obstacle scenarios where the baseline planner suffers repeated collisions.
2. **Hypothesis 2 (Kinetic Shock Reduction):** Under sudden dynamic intrusions where an emergency stop is inevitable, the predictive monitor triggers deceleration $0.10\text{--}0.50\,\text{s}$ earlier than reactive monitoring, significantly reducing kinetic energy at stopping and eliminating physical impact force.
3. **Hypothesis 3 (Task Efficiency & Non-Intrusiveness):** On clean baseline episodes and hard negative boundary-grazing scenarios, the predictive safety system maintains a **False Intervention Rate $\le 5\%$**, preserving nominal task completion times ($T_{\text{comp}} \approx 5.1\,\text{s}$).

---

## 2. Experimental Isolation & Industrial Safety Clarification

### 2.1 Strict Isolation of Configuration C (Predictive Only)
- **Scientific Role:** Configuration C (Predictive Intervention Only) exists **strictly as an offline simulation ablation tool**. Its sole purpose is scientific attribution: quantifying what proportion of collisions the world model can avert purely via lookahead when the reactive contact latch is disabled.
- **Strict Prohibition:** Configuration C is **strictly prohibited from being used as an operational or default controller**.
- **Default Operational Controller:** The default controller for SAWM is **Configuration D (Combined Defense-in-Depth)**.

### 2.2 Clarification on Simulated Reactive Safety vs Certified Industrial Systems
- **Simulation Research Scope:** SAWM's Layer 0 reactive baseline is a **simulated research model** implemented within the MuJoCo physics engine via contact pair evaluation (`env.data.ncon`) and distance thresholds.
- **Not a Certified Safety System:** It is **not** a certified functional safety system (such as ISO 13849-1 PLd/PLe or ISO 10218-1/2 safety-rated monitored stops or dual-channel safety PLCs).
- **Disclaimer:** SAWM makes no sim-to-real claims or industrial certification claims; it is an academic algorithmic research framework for safety-aware robotics.

---

## 3. Experimental Configurations (4-Way Comparative Study)

```
+--------------------------------------------------------------------------------------------------+
|                                4-WAY COMPARATIVE CONFIGURATIONS                                  |
+--------------------------------------------------------------------------------------------------+
| Configuration A: Baseline Planner (No Safety Intervention)                                       |
| - Standard pick-and-place planner operating open-loop with respect to hazards.                  |
| - Safety heads observe passively (M3/M4 baseline).                                               |
+--------------------------------------------------------------------------------------------------+
| Configuration B: Reactive Safety Only (Conventional Industrial Baseline in Sim)                  |
| - Layer 0 active: Robot triggers EMERGENCY_STOP only upon physical contact (c_haz == 1) or       |
|   immediate distance breach (d_geom <= 0.01m).                                                   |
| - Zero predictive lookahead or world model guidance.                                             |
+--------------------------------------------------------------------------------------------------+
| Configuration C: Predictive Intervention Only (STRICT OFFLINE ABLATION ONLY)                     |
| - Layers 1 & 2 active: Safety Supervisor modulates velocity and replans based on RSSM           |
|   tactical (H=10 at 25 Hz) and strategic (H=50 at 10 Hz) heads.                                  |
| - Layer 0 reactive contact latch artificially disabled to evaluate pure predictive efficacy.     |
| - PROHIBITED AS OPERATIONAL CONTROLLER.                                                          |
+--------------------------------------------------------------------------------------------------+
| Configuration D: Combined Defense-in-Depth (Full SAWM Default Architecture)                      |
| - Full multi-rate architecture: Layer 0 Reactive + Layer 1 Tactical (25 Hz) + Layer 2 Strategic. |
| - Predictive monitors proactively slow down and replan; reactive monitor provides final veto.   |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. Evaluation Scenarios & Test Suite

The evaluation suite encompasses 5 distinct scenarios, tested across $N = 20$ randomized seeds per scenario ($100$ runs per configuration, **$400$ total evaluation runs**):

### Scenario 1: Clean Baseline Pick-and-Place (Nominal Operation)
- **Goal:** Verify zero false-alarm interventions on nominal tasks.
- **Setup:** No obstacles. Target object spawned at randomized table locations ($x \in [0.45, 0.60]$, $y \in [-0.20, 0.20]$).
- **Key Metrics:** Task Success Rate ($TSR \ge 95\%$), False Intervention Rate ($FIR \le 5\%$), Completion Time.

### Scenario 2: Static Pillar in Transit Corridor
- **Goal:** Evaluate proactive spatial bypass and early slowdown around unmodeled static obstacles.
- **Setup:** Cylindrical pillar ($r = 0.04\,\text{m}, h = 0.50\,\text{m}$) placed along the direct pick-to-place transit line.
- **Key Metrics:** Collision Rate ($C_R$), Minimum Clearance ($d_{\min} \ge 0.05\,\text{m}$), Detour Success Rate.

### Scenario 3: Place Descent Obstacle
- **Goal:** Evaluate vertical approach safety where an obstacle partially obstructs the target placement fixture.
- **Setup:** Low barrier placed within $80\,\text{mm}$ of the placement goal.
- **Key Metrics:** Collision Rate ($C_R$), Controlled Halt before contact, Absence of table impact.

### Scenario 4: Dynamic Human-Proxy Intrusion
- **Goal:** Test high-speed tactical early warning and collision avoidance against moving hazards.
- **Setup:** Capsule human proxy enters workspace, traversing the transit corridor at three distinct velocities:
  - Slow intrusion: $v = 0.2\,\text{m/s}$
  - Moderate intrusion: $v = 0.5\,\text{m/s}$
  - Rapid crossing: $v = 0.8\,\text{m/s}$
- **Key Metrics:** Collision Rate, Stopping Distance to Proxy, Lead Time, Kinetic Energy at Arrest.

### Scenario 5: Boundary Grazing / Hard Negatives
- **Goal:** Stress-test false-positive rejection under high geometric proximity.
- **Setup:** Static obstacle or moving proxy passes within $0.09\text{--}0.15\,\text{m}$ of the planned path without intersecting it.
- **Key Metrics:** False Intervention Rate ($FIR$), Unnecessary Slowdown Count, Path Delay.

---

## 5. Quantitative Metrics and Definitions

```
+--------------------------+-------------------------------------------------------+-----------------------+
| Metric Symbol            | Metric Name & Mathematical Formulation                | Target / Good Value   |
+--------------------------+-------------------------------------------------------+-----------------------+
| C_R                      | Collision Rate: N_collisions / N_trials               | 0.0%                  |
| N_R                      | Near-Miss Rate: N(0 < d_geom <= 0.08m) / N_trials    | Minimize (< 10%)      |
| d_min                    | Minimum Geometric Clearance across trajectory (mm)   | >= 50.0 mm            |
| FIR                      | False Intervention Rate on Scenarios 1 & 5            | <= 5.0%               |
| TSR                      | Task Success Rate: Object placed within 15mm of goal  | >= 90.0%              |
| T_comp                   | Task Completion Time (seconds)                        | <= 6.5s (clean = 5.1s)|
| L_traj                   | Total End-Effector Trajectory Path Length (m)         | Minimal increase (<15%)|
| T_latency                | Lead Time: Time from first alarm to would-be collision| >= 0.10s (Tactical)   |
| RSR                      | Recovery Success Rate: Resumed tasks reaching goal    | >= 85.0%              |
| DAR                      | Detour Acceptance Rate (Passed all 7 gates)           | >= 80% (when feasible)|
+--------------------------+-------------------------------------------------------+-----------------------+
```

---

## 6. Targeted Ablation Studies

To isolate the individual contribution of each sub-component, we conduct five targeted ablation studies:

```
Ablation Matrix
---------------------------------------------------------------------------------------------------
Ablation ID   Tactical Head (H10)   Strategic Head (H50)   Intervention Mode    Filtering Mode
---------------------------------------------------------------------------------------------------
Ablation-1    Active (25 Hz)        Disabled               Controlled Decel     Dual-Tier (2-of-3)
Ablation-2    Disabled              Active (10 Hz Chunked) 7-Gate Detour Replan Dual-Tier (2-of-3)
Ablation-3    Active (25 Hz)        Active (10 Hz Chunked) Controlled Decel     Dual-Tier (2-of-3)
Ablation-4    Active (25 Hz)        Active (10 Hz Chunked) Hybrid (Decel+Replan)Raw (No Filter)
Ablation-5    Active (33 Hz Opt)    Active (10 Hz Chunked) Hybrid (Decel+Replan)Dual-Tier (2-of-3)
Full System   Active (25 Hz Base)   Active (10 Hz Chunked) Hybrid (Decel+Replan)Dual-Tier Filter
---------------------------------------------------------------------------------------------------
```

---

## 7. Statistical Rigor, Latency Profiling & Repeatability Protocol

1. **Formal 1,000-Cycle Latency Profiling Protocol:**
   - Execute at least **1,000 continuous tactical cycles** under concurrent simulation stepping and strategic background operations.
   - Report p50, p95, p99, p99.9, absolute maximum latency, and the exact count and percentage of cycles exceeding the $40.0\,\text{ms}$ deadline.
2. **Planned Concurrency Verification Suite:**
   - The planned Milestone 6 test suite (`tests/test_predictive_monitors_concurrency.py` — an unbuilt, planned M6 deliverable) will execute the final concurrency benchmark measuring H10 end-to-end latency while H50 is executing, with identical measurement boundaries and explicit synchronization.
3. **Deterministic Random Seed Matrix:**
   - Evaluated across seeds `1001` through `1020` for every scenario and configuration.
   - Initial object position, obstacle offset, and proxy velocity are generated deterministically per seed.
4. **Confidence Intervals:**
   - All binomial rates ($C_R, TSR, FIR, RSR, DAR$) are reported with exact 95% Wilson Score intervals:
     $$p_{\text{Wilson}} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.96)$$
5. **Continuous Distribution Statistics:**
   - Clearances and completion times reported as Mean, Median, Std, p05, and p95 to capture worst-case tail performance.
6. **Execution Artifacts:**
   - Trajectory telemetry logged per trial: time-synchronized array of $(t, q, \dot{q}, s_v, \text{State}, \hat{P}_{10}, \hat{d}_{10}, \hat{P}_{50}, \hat{d}_{50}, d_{\text{geom}}, c_{\text{haz}})$.
   - Saved in `results/m6_eval_telemetry/` as NPZ files.
   - Aggregated metrics compiled in `results/milestone6_evaluation_report.json` and human-readable markdown.

---

## 8. Document Revision Log

| Revision | Date | Author | Summary of Changes and Technical Rationale |
|---|---|---|---|
| **v1.0** | Nov 15, 2026 | Antigravity | Initial experimental plan. |
| **v2.0** | Nov 15, 2026 | Antigravity | Added Configuration C isolation, industrial safety disclaimer, and initial ablation matrix. |
| **v2.1** | Nov 15, 2026 | Antigravity | Refined baseline frequency to 25 Hz and updated stopping definitions. |
| **v2.2** | Nov 15, 2026 | Antigravity | Added formal 1,000-cycle latency benchmarking requirement. |
| **v2.3** | Nov 15, 2026 | Antigravity | **Final consistency check corrections:**<br>1. **Tightened Benchmark Terminology:** Labeled existing benchmarks as preliminary. Explicitly required final concurrency benchmark to measure H10 end-to-end latency while H50 is executing, with identical measurement boundaries and explicit synchronization.<br>2. **Planned Concurrency Suite:** Clarified that `tests/test_predictive_monitors_concurrency.py` is an unbuilt, planned M6 test suite.<br>3. **Simulated Stop Metrics:** Aligned stopping performance metrics with the corrected 1D kinematic estimate and actuator model constraints.<br>4. **Degraded Recovery Tracking:** Explicitly added tracking of degraded recovery modes when H50 is unavailable. |
