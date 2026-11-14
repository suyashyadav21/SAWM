# SAWM: Safety-Aware World Model for Franka Panda Robotics

A research-oriented prototype investigating safety-aware predictive world modeling, risk estimation, and counterfactual planning for robotic manipulation.

---

## Milestone 1: Minimal Franka Panda Robot Simulation (COMPLETED)

### Overview
Milestone 1 establishes the core physical simulation foundation:
- **Physics Engine:** MuJoCo 4.0.1 (native Python bindings)
- **Robot Model:** Franka Emika Panda (7-DOF arm + 2-finger parallel gripper) obtained from official `google-deepmind/mujoco_menagerie` under Apache-2.0 license
- **Simulation Scene:** Workstation table, movable target cube with realistic contact dynamics, visual goal zone, and multi-angle observation cameras
- **Robot Controller:** Position-actuated joint controller with hardware joint limit enforcement and smooth cosine-interpolated trajectory generation
- **Safety Foundations:** Real-time contact and collision detection, joint limit monitoring, penetration tracking
- **Rendering:** Multi-camera offscreen rendering (1280×720 RGB) and optional interactive 3D viewer

---

## Project Structure

```
SAWM/
├── .venv/                         # Python 3.11.9 virtual environment
├── configs/
│   └── simulation.yaml            # Environment, robot limits, and render settings
├── assets/
│   └── robots/
│       └── franka_emika_panda/    # Official MuJoCo Menagerie Franka model & meshes
│           ├── assets/            # Link & finger OBJ/STL meshes
│           ├── panda.xml          # Robot definition (Apache-2.0)
│           ├── sawm_arena.xml     # Simulation arena (table, object, cameras)
│           └── LICENSE            # Apache-2.0 License
├── simulation/
│   ├── __init__.py
│   ├── robot.py                   # FrankaPandaRobot controller & state interface
│   ├── environment.py             # SawmEnvironment MuJoCo wrapper
│   └── utils.py                   # Pure-Python PNG image writer & helpers
├── scripts/
│   └── run_simulation.py          # Demonstration execution script
├── tests/
│   └── test_simulation.py         # Automated test suite (5 test modules)
├── results/
│   └── milestone1_simulation.png  # Rendered simulation screenshot (1280x720)
├── requirements.txt               # Installed dependencies
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

### 1. Run Automated Test Suite
Executes all 5 test modules covering environment initialization, joint limits, trajectory execution, contact detection, and offscreen rendering:

```powershell
.\.venv\Scripts\python.exe tests\test_simulation.py
```

### 2. Run Simulation Demonstration Script
Runs a multi-stage trajectory (Home -> Ready -> Tabletop descent -> Grasp -> Release -> Retract) with real-time terminal telemetry and captures a high-resolution screenshot:

```powershell
.\.venv\Scripts\python.exe scripts\run_simulation.py
```

### 3. Run with Interactive 3D Viewer (Desktop Display)
Launches the interactive native MuJoCo passive viewer:

```powershell
.\.venv\Scripts\python.exe scripts\run_simulation.py --view
```

---

## Milestone 1 Verification Results

| Deliverable | Status | Verification Detail |
|---|---|---|
| Python 3.11.9 Virtual Environment | ✅ Complete | Verified at `C:\Projects\SAWM\.venv` |
| MuJoCo 4.0.1 + NumPy + PyYAML | ✅ Complete | Successfully installed & imported |
| Franka Panda Model from Menagerie | ✅ Complete | Downloaded with meshes & Apache-2.0 license |
| Arena Scene (Table + Target Object) | ✅ Complete | `sawm_arena.xml` with table, cube, goal, 3 cameras |
| Robot Joint Control & Kinematics | ✅ Complete | 7 arm actuators + gripper, cosine interpolation |
| Joint Limit Verification | ✅ Complete | Hardware limits monitored on all 7 joints |
| Collision & Contact Detection | ✅ Complete | Object-table & robot-table contact querying |
| Multi-Camera Offscreen Rendering | ✅ Complete | 1280×720 framebuffer rendering via `mujoco.Renderer` |
| Saved Screenshot | ✅ Complete | `results/milestone1_simulation.png` (156 KB) |
| Automated Test Suite | ✅ Complete | 5/5 tests passing |
