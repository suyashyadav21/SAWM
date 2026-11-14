# SAWM: Safety-Aware World Model for Franka Panda Robotics

A research-oriented prototype investigating safety-aware predictive world modeling, risk estimation, and counterfactual planning for robotic manipulation.

---

## Milestone Status Overview

- **Milestone 0 (Audit & Research):** ✅ Complete
- **Milestone 1 (Minimal Robot Simulation):** ✅ Complete
- **Milestone 2 (Hazard & Safety Scenarios):** ✅ **Complete & Verified**
- **Milestone 3 (Baseline Planner):** ⏳ Next
- **Milestone 4 (Predictive World Model):** ⏳ Pending
- **Milestone 5 (Safety Modules):** ⏳ Pending
- **Milestone 6 (Counterfactual Safety Planning):** ⏳ Pending
- **Milestone 7 (Interactive Dashboard):** ⏳ Pending
- **Milestone 8 (Final Validation & Comparative Metrics):** ⏳ Pending

---

## Milestone 2: Hazard & Safety Scenarios (COMPLETED)

### Overview
Milestone 2 establishes the physical hazard, contextual zone, and centralized safety monitoring system:
- **11-Body Robot Kinematic Tracking:** Explicitly monitors all 11 robot bodies (`link0`–`link7`, `hand`, `left_finger`, `right_finger`) and 23 active collision geoms.
- **Physical Hazards:** Static rigid obstacle and dynamic moving hazard proxy (modeled as a simplified kinematic arm/hand representation using MuJoCo mocap bodies).
- **Surface-to-Surface Distance Engine:** MuJoCo GJK/EPA solver (`mujoco.mj_geomDistance`) computes exact signed Euclidean distances, cleanly distinguished from approximate link-center clearances.
- **Contextual Keep-Out Zones:** Evaluates 3D Cartesian world coordinates (`data.xpos`, `data.geom_xpos`) against restricted workspace bounding volumes. Joint angles (`qpos`) are strictly excluded from spatial checks.
- **Centralized Safety Monitor:** 3-tier state machine (`SAFE`, `WARNING`, `DANGER`):
  - Proximity warnings recover to `SAFE` automatically once hazards clear.
  - Physical collision events trigger **latched DANGER**, locking the state until an explicit `reset()`.
- **5 Verified Scenarios:** Fully reproducible demonstration of normal operation, proximity warning, zone breach, danger near-miss, and collision latching.

---

## Project Structure

```
SAWM/
├── .venv/                         # Python 3.11.9 virtual environment
├── configs/
│   ├── simulation.yaml            # Environment, robot limits, and render settings
│   └── safety.yaml                # Experimental safety thresholds, hazard coords, scenarios
├── assets/
│   └── robots/
│       └── franka_emika_panda/    # Official MuJoCo Menagerie Franka model & meshes
│           ├── assets/            # Link & finger OBJ/STL meshes
│           ├── panda.xml          # Robot definition (Apache-2.0)
│           ├── sawm_arena.xml     # Arena with hazards, mocap proxies, and zones
│           └── LICENSE            # Apache-2.0 License
├── observation/
│   ├── state.py                   # Structured schemas (RobotState, DistanceMetrics, etc.)
│   └── observation_manager.py     # 11-body tracker, mj_geomDistance, Cartesian zone evaluator
├── safety/
│   ├── __init__.py
│   ├── physical_risk.py           # Proximity & contact evaluation
│   ├── contextual_harm.py         # Restricted zone intrusion evaluation
│   └── safety_monitor.py          # 3-tier state machine with latched collision DANGER
├── simulation/
│   ├── __init__.py
│   ├── robot.py                   # FrankaPandaRobot controller & state interface
│   ├── environment.py             # SawmEnvironment MuJoCo wrapper
│   ├── obstacles.py               # ObstacleManager (mocap body controls)
│   ├── scenarios.py               # ScenarioManager (orchestrates 5 safety scenarios)
│   └── utils.py                   # Pure-Python PNG image writer & helpers
├── scripts/
│   ├── run_simulation.py          # Milestone 1 simulation runner
│   └── run_scenarios.py           # Milestone 2 scenario demonstration runner
├── tests/
│   ├── test_simulation.py         # Milestone 1 test suite (5 tests)
│   └── test_hazards.py            # Milestone 2 test suite (6 tests)
├── results/                       # High-resolution demonstration screenshots
│   ├── milestone1_simulation.png
│   ├── milestone2_scenario1.png
│   ├── milestone2_scenario2.png
│   ├── milestone2_scenario3.png
│   ├── milestone2_scenario4.png
│   └── milestone2_scenario5.png
├── requirements.txt               # Minimal pinned dependencies
└── README.md
```

---

## Verified Environment

- **OS:** Windows 11 (64-bit)
- **Python:** 3.11.9 (`C:\Projects\SAWM\.venv\Scripts\python.exe`)
- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop (4096 MiB VRAM)
- **Dependencies (Minimal):**
  - `mujoco==4.0.1`
  - `numpy==2.4.6`
  - `pyyaml==6.0.3`

---

## Execution Commands

### 1. Run Automated Test Suites

**Run Milestone 1 Tests (Verification of zero regressions):**
```powershell
.\.venv\Scripts\python.exe tests\test_simulation.py
```

**Run Milestone 2 Tests (Hazards & Safety Scenarios):**
```powershell
.\.venv\Scripts\python.exe tests\test_hazards.py
```

### 2. Run Scenario Demonstration Scripts

**Run all 5 scenarios sequentially:**
```powershell
.\.venv\Scripts\python.exe scripts\run_scenarios.py --all
```

**Run an individual scenario (e.g. Scenario 2 - Obstacle Near Path):**
```powershell
.\.venv\Scripts\python.exe scripts\run_scenarios.py --scenario 2
```

**Run with interactive 3D viewer (on desktop display):**
```powershell
.\.venv\Scripts\python.exe scripts\run_scenarios.py --scenario 4 --view
```

---

## Milestone 2 Scenario Summary & Acceptance Criteria

| Scenario | Description | Key Metric | Result | Status |
|---|---|---|---|---|
| **1. Normal Operation** | Clear workspace | $d_{\min} > 0.25\,\text{m}$, 0 near-misses, 0 collisions | Stays `SAFE` throughout | ✅ PASS |
| **2. Obstacle Near Path** | Static pillar 0.12m from corridor | $0.08\,\text{m} \le d_{\min} < 0.20\,\text{m}$, 0 collisions | `SAFE` $\rightarrow$ `WARNING` $\rightarrow$ `SAFE` | ✅ PASS |
| **3. Restricted Zone** | Trajectory enters keep-out box | $d_{\text{zone}} = 0\,\text{m}$, collision = False | `ZONE_BREACH` $\rightarrow$ `DANGER` | ✅ PASS |
| **4. Moving Kinematic Proxy** | Arm proxy approaches $d = 0.039\,\text{m}$ | $d_{\min} < 0.08\,\text{m}$, near-miss = 1, 0 collisions | `DANGER` $\rightarrow$ recovers to `SAFE` | ✅ PASS |
| **5. Physical Collision Event** | Obstacle placed directly in path | $con.dist < 0$, collision = 1 | `DANGER` (Latched = True) | ✅ PASS |
