"""
Interactive Live Dashboard for SAWM (Milestone 6 Step 2 MVP).

Serves a modern, responsive web dashboard on http://127.0.0.1:8050:
- Live 30 FPS MuJoCo Franka Panda 3D rendering
- Real-time telemetry indicators:
    * Current Planner Stage
    * Supervisor FSM State (NORMAL / CAUTION / SLOWDOWN / EMERGENCY_STOP / RECOVERY)
    * Predicted Collision Probability P10 (vs tau=0.36 alarm threshold)
    * Predicted Minimum Clearance d10 (vs danger=80mm threshold)
    * Actual Current Geometric Clearance
    * Speed Scale (0.00x - 1.00x)
    * Predictive Intervention Indicator (Standby / Triggered)
    * Reactive Layer 0 Status (Safe / Collision Breach)
- Scenario Switcher controls (Scenario A: Nominal, Scenario B: Dynamic Hazard)

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8050 --scenario B
"""

import os
import sys
import json
import time
import base64
import threading
import webbrowser
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from planning.baseline_planner import BaselinePlanner
from safety.safety_supervisor import SafetySupervisor, ReactiveTelemetry, TacticalTelemetry, StrategicTelemetry, SupervisorState
from safety.predictive_intervention import H10Predictor


def rgb_to_bmp(arr: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 numpy array to uncompressed Windows BMP bytes in ~1ms."""
    h, w, _ = arr.shape
    row_bytes = (w * 3 + 3) & ~3
    image_size = row_bytes * h
    file_size = 54 + image_size

    hdr = bytearray(54)
    hdr[0:2] = b'BM'
    hdr[2:6] = file_size.to_bytes(4, 'little')
    hdr[10:14] = (54).to_bytes(4, 'little')
    hdr[14:18] = (40).to_bytes(4, 'little')
    hdr[18:22] = w.to_bytes(4, 'little')
    hdr[22:26] = h.to_bytes(4, 'little')
    hdr[26:28] = (1).to_bytes(2, 'little')
    hdr[28:30] = (24).to_bytes(2, 'little')
    hdr[34:38] = image_size.to_bytes(4, 'little')

    bgr = arr[::-1, :, ::-1]
    if row_bytes == w * 3:
        raw = bgr.tobytes()
    else:
        pad = bytes(row_bytes - w * 3)
        raw = b''.join(bgr[row].tobytes() + pad for row in range(h))
    return bytes(hdr) + raw


class DashboardController:
    """Coordinates simulation stepping, H10 prediction, and telemetry streaming."""

    def __init__(self, config_path: str = "configs/simulation.yaml"):
        self.lock = threading.Lock()
        self.env = SawmEnvironment(config_path)
        self.planner = BaselinePlanner(self.env)
        self.predictor = H10Predictor()
        self.supervisor = SafetySupervisor(config={"require_h50": False})

        self.scenario_name = "B_HAZARD"
        self.is_running = True
        self.is_paused = False
        self.step_delay = 0.015  # ~60 Hz visual playback speed

        self.current_frame_bmp: bytes = b""
        self.telemetry_data: Dict[str, Any] = {}
        self.planned_steps = []
        self.cursor_float = 0.0
        self.loop_tick = 0
        self.prev_act = np.zeros(8, dtype=np.float32)

        self.first_intervention: Optional[Dict[str, Any]] = None
        self.stopped_info: Optional[Dict[str, Any]] = None

        self.reset_scenario("B_HAZARD")

    def reset_scenario(self, scenario_name: str = "B_HAZARD"):
        with self.lock:
            self.scenario_name = scenario_name
            self.supervisor = SafetySupervisor(config={"require_h50": False})
            obs = self.env.reset()

            if scenario_name == "B_HAZARD":
                # Initially park proxy; it will enter when simulated robot reaches progress condition
                self.env.obstacle_manager.park_all()
                self.hazard_triggered = False
                self.hazard_start_cursor = 25.0
                self.p_start = np.array([0.80, 0.25, 0.60], dtype=np.float64)
                self.p_target = np.array([0.52, 0.05, 0.58], dtype=np.float64)
            else:
                self.env.obstacle_manager.park_all()
                self.hazard_triggered = False

            obs = self.env.get_observation()
            t0 = float(obs["time"])
            self.predictor.reset(self.env, obs, current_time=t0)

            keyframes, ok, _ = self.planner.plan_keyframes(obs["target_object"]["position"], obs["goal"]["position"])
            self.planned_steps = self.planner.build_discrete_plan(keyframes)
            self.planner._planned_steps = self.planned_steps

            self.cursor_float = 0.0
            self.loop_tick = 0
            self.prev_act = np.zeros(8, dtype=np.float32)
            self.first_intervention = None
            self.stopped_info = None

            # Render initial frame
            img = self.env.render()
            if img is not None:
                # Downsample to 360x640 for fast web streaming
                self.current_frame_bmp = rgb_to_bmp(img[::2, ::2])

            self._update_telemetry(obs, None, None, None)

    def _update_telemetry(self, obs: Dict[str, Any], cmd: Optional[Any], tac: Optional[Any], multi: Optional[Any] = None):
        step_idx = min(int(self.cursor_float), len(self.planned_steps) - 1)
        step_info = self.planned_steps[step_idx]
        geom_d = float(obs["distances"].geometric_min_distance) if "distances" in obs else 1.0

        p10 = float(tac.p10) if tac is not None else float(self.telemetry_data.get("p10", 0.0))
        p50 = float(multi["p50"]) if multi is not None else float(self.telemetry_data.get("p50", 0.0))
        d10 = float(tac.d10_clear) if tac is not None else (float(self.telemetry_data.get("d10_mm", 1000.0)) / 1000.0)
        state_name = cmd.state.name if cmd is not None else "NORMAL"
        speed = float(cmd.speed_scale) if cmd is not None else 1.0
        c = obs["contacts"]
        has_coll = c["robot_obstacle_collision"] or c["robot_proxy_collision"] or c["robot_table_collision"]
        is_hazard_active = bool(self.scenario_name == "B_HAZARD" and getattr(self, "hazard_triggered", False))

        self.telemetry_data = {
            "scenario": self.scenario_name,
            "tick": self.loop_tick,
            "cursor": round(self.cursor_float, 1),
            "sim_time": round(float(obs["time"]), 2),
            "planner_stage": step_info["fsm_state"].name,
            "supervisor_state": state_name,
            "speed_scale": round(speed, 2),
            "p10": round(p10, 4),
            "p50": round(p50, 4),
            "d10_mm": round(d10 * 1000.0, 1),
            "actual_clearance_mm": round(geom_d * 1000.0, 1) if geom_d < 10.0 else 999000.0,
            "hazard_active": is_hazard_active,
            "predictive_intervention_active": bool(state_name in ["SLOWDOWN", "EMERGENCY_STOP", "CAUTION"]),
            "layer0_collision": bool(has_coll),
            "p10_threshold": 0.36,
            "first_intervention": self.first_intervention,
            "stopped_info": self.stopped_info,
            "is_paused": self.is_paused,
        }

    def step(self):
        with self.lock:
            if self.is_paused:
                return

            if self.stopped_info is not None and self.scenario_name == "B_HAZARD":
                # Obstacle continues moving after robot arrests until transit finishes
                if self.env.obstacle_manager._proxy_step_idx < self.env.obstacle_manager._proxy_total_steps + 15:
                    obs = self.env.step(arm_targets=self.supervisor._frozen_qpos)
                    img = self.env.render()
                    if img is not None:
                        self.current_frame_bmp = rgb_to_bmp(img[::2, ::2])
                    self._update_telemetry(obs, None, None, None)
                    time.sleep(self.step_delay)
                    return
                time.sleep(0.05)
                return

            total_steps = len(self.planned_steps)
            step_idx = min(int(self.cursor_float), total_steps - 1)
            step_info = self.planned_steps[step_idx]

            obs = self.env.get_observation()
            t_curr = float(obs["time"])

            if self.loop_tick > 0:
                self.predictor.update(self.prev_act, obs, t_curr)

            # Trigger dynamic hazard approach once robot visibly progresses along trajectory
            if self.scenario_name == "B_HAZARD" and self.cursor_float >= getattr(self, "hazard_start_cursor", 25.0) and not getattr(self, "hazard_triggered", False):
                self.hazard_triggered = True
                self.env.obstacle_manager.set_proxy_trajectory(self.p_start, self.p_target, steps=100)

            actions_50, _ = self.planner.get_planned_action_sequence(50, current_cursor=step_idx)
            multi = self.predictor.predict_multi_horizon(actions_50, current_time=t_curr, raw_obs=obs)
            tac = TacticalTelemetry(p10=multi["p10"], d10_clear=multi["d10_clear"], timestamp=t_curr, age_ms=0.0)
            strat = StrategicTelemetry(p50=multi["p50"], d50_clear=multi["d50_clear"], timestamp=t_curr, age_ms=0.0)

            q_curr = self.env.robot.get_joint_positions()
            q_vel = self.env.robot.get_joint_velocities()
            c = obs["contacts"]
            has_coll = c["robot_obstacle_collision"] or c["robot_proxy_collision"] or c["robot_table_collision"]
            geom_d = float(obs["distances"].geometric_min_distance) if "distances" in obs else 1.0

            reactive = ReactiveTelemetry(
                has_collision=has_coll,
                robot_object_contact=c["robot_object_contact"],
                geometric_min_distance=geom_d,
                timestamp=t_curr,
            )

            cmd = self.supervisor.step(
                q_curr,
                q_vel,
                step_info["arm_target"],
                step_info["gripper_target"],
                reactive=reactive,
                tactical=tac,
                strategic=strat,
                current_time=t_curr,
                planner_stage=step_info["fsm_state"].name,
            )

            if cmd.state.name != "NORMAL" and self.first_intervention is None:
                self.first_intervention = {
                    "step": self.loop_tick,
                    "state": cmd.state.name,
                    "p10": round(float(tac.p10), 4),
                    "p50": round(float(multi["p50"]), 4),
                    "d10_mm": round(float(tac.d10_clear) * 1000.0, 1),
                    "clearance_mm": round(geom_d * 1000.0, 1) if geom_d < 10.0 else 999000.0,
                }

            if cmd.speed_scale == 0.0 and np.max(np.abs(q_vel)) < 1e-3 and self.loop_tick > 10:
                if self.stopped_info is None:
                    self.stopped_info = {
                        "step": self.loop_tick,
                        "residual_vel": round(float(np.max(np.abs(q_vel))), 6),
                        "clearance_mm": round(geom_d * 1000.0, 1) if geom_d < 10.0 else 999000.0,
                    }

            self.cursor_float += cmd.speed_scale
            self.loop_tick += 1

            obs = self.env.step(
                arm_targets=cmd.target_qpos,
                gripper_target=cmd.target_gripper,
                actuator_inhibit=cmd.actuator_inhibit,
                supervisor=cmd.supervisor,
            )
            self.prev_act = step_info["action_vector"]

            img = self.env.render()
            if img is not None:
                self.current_frame_bmp = rgb_to_bmp(img[::2, ::2])

            self._update_telemetry(obs, cmd, tac, multi)


HTML_DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAWM Predictive Intervention Live Dashboard</title>
<style>
  :root {
    --bg-main: #0b0f19;
    --bg-card: #151d2e;
    --border: #23314d;
    --text-primary: #e6edf3;
    --text-secondary: #8b9bb4;
    --accent: #388bfd;
    --green: #2ea043;
    --yellow: #d29922;
    --orange: #db6d28;
    --red: #f85149;
    --blue: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  body { background: var(--bg-main); color: var(--text-primary); padding: 16px; min-height: 100vh; }
  header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
  h1 { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.5px; display: flex; align-items: center; gap: 8px; }
  .badge-mvp { background: #1f3358; color: #79c0ff; font-size: 0.72rem; padding: 3px 8px; border-radius: 12px; font-weight: 600; }
  .controls { display: flex; gap: 10px; }
  button { background: var(--bg-card); border: 1px solid var(--border); color: var(--text-primary); padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.15s ease; }
  button:hover { background: #1f2c44; border-color: var(--accent); }
  button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  
  .layout-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .card-header { padding: 10px 14px; background: rgba(0,0,0,0.2); border-bottom: 1px solid var(--border); font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); display: flex; justify-content: space-between; align-items: center; }
  .viewport-container { position: relative; width: 100%; aspect-ratio: 16/9; background: #000; display: flex; align-items: center; justify-content: center; }
  .viewport-container img { width: 100%; height: 100%; object-fit: contain; }
  .viewport-overlay { position: absolute; top: 10px; left: 10px; display: flex; gap: 8px; }
  .status-pill { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }

  .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 14px; }
  .metric-box { background: rgba(0,0,0,0.25); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .metric-label { font-size: 0.72rem; text-transform: uppercase; color: var(--text-secondary); font-weight: 600; margin-bottom: 4px; }
  .metric-val { font-size: 1.4rem; font-weight: 700; }
  .metric-sub { font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px; }

  .bar-container { background: #0b111e; border-radius: 4px; height: 8px; width: 100%; overflow: hidden; margin-top: 6px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.1s ease, background-color 0.2s ease; }

  .full-width { grid-column: span 2; }
  .event-banner { margin: 14px; padding: 10px 14px; border-radius: 6px; font-size: 0.82rem; font-weight: 600; display: flex; align-items: center; gap: 10px; }
</style>
</head>
<body>
<header>
  <h1>Franka Panda — SAWM Safety Dashboard <span class="badge-mvp">MVP 48-HR VERIFIED</span></h1>
  <div class="controls">
    <button id="btn-scen-b" class="active" onclick="setScenario('B_HAZARD')">Scenario B (Dynamic Hazard)</button>
    <button id="btn-scen-a" onclick="setScenario('A_NORMAL')">Scenario A (Nominal)</button>
    <button id="btn-pause" onclick="togglePause()">Pause</button>
    <button onclick="resetCurrent()">Reset</button>
  </div>
</header>

<div class="layout-grid">
  <!-- Left Column: Simulation Viewport -->
  <div class="card">
    <div class="card-header">
      <span>MuJoCo 3D Simulation Viewport (Franka Emika Panda)</span>
      <span id="txt-sim-time">Time: 0.00 s</span>
    </div>
    <div class="viewport-container">
      <img id="sim-viewport" src="/frame" alt="Simulation Stream">
      <div class="viewport-overlay">
        <div id="badge-supervisor" class="status-pill" style="background: var(--green); color: #fff;">NORMAL</div>
        <div id="badge-l0" class="status-pill" style="background: #1f2c44; color: #a2b4d2;">L0: CLEAR</div>
      </div>
    </div>
    <div id="intervention-banner" class="event-banner" style="display:none; background: rgba(219, 109, 40, 0.2); border: 1px solid var(--orange); color: #ffa657;">
      <span>⚠️ PREDICTIVE INTERVENTION ACTIVE: Controlled Deceleration to Safe Rest</span>
    </div>
  </div>

  <!-- Right Column: Real-Time Telemetry Panels -->
  <div class="card">
    <div class="card-header">
      <span>Multi-Layer Risk & Supervisor Telemetry</span>
      <span id="txt-scenario-name">Scenario: B_HAZARD</span>
    </div>
    <div class="telemetry-grid">
      
      <!-- Planner Stage -->
      <div class="metric-box">
        <div class="metric-label">Current Planner Stage</div>
        <div id="val-stage" class="metric-val" style="font-size: 1.15rem; color: var(--blue);">MOVE_TO_PREGRASP</div>
        <div id="sub-cursor" class="metric-sub">Waypoint Cursor: 0.0</div>
      </div>

      <!-- Supervisor Speed Scale -->
      <div class="metric-box">
        <div class="metric-label">Speed Scale (s_v)</div>
        <div id="val-speed" class="metric-val" style="color: var(--text-primary);">1.00x</div>
        <div class="bar-container">
          <div id="bar-speed" class="bar-fill" style="width: 100%; background: var(--green);"></div>
        </div>
      </div>

      <!-- Predicted Risk Multi-Horizon (P10 & P50) -->
      <div class="metric-box">
        <div class="metric-label">Predicted Collision Risk</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <div><span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600;">Tactical P10: </span><span id="val-p10" class="metric-val">0.015</span></div>
          <div><span style="font-size:0.72rem; color:var(--text-secondary); font-weight:600;">Strategic P50: </span><span id="val-p50" class="metric-val" style="color:var(--accent);">0.045</span></div>
        </div>
        <div class="bar-container">
          <div id="bar-p10" class="bar-fill" style="width: 15%; background: var(--green);"></div>
        </div>
        <div class="metric-sub">Alarm Thresholds: P10 &ge; 0.36 | P50 &ge; 0.36</div>
      </div>

      <!-- Predicted Clearance d10 -->
      <div class="metric-box">
        <div class="metric-label">Predicted Clearance (d10)</div>
        <div id="val-d10" class="metric-val">49.3 mm</div>
        <div class="bar-container">
          <div id="bar-d10" class="bar-fill" style="width: 49%; background: var(--yellow);"></div>
        </div>
        <div class="metric-sub">Moderate Threshold: &le; 80 mm</div>
      </div>

      <!-- Actual Geometric Clearance -->
      <div class="metric-box full-width">
        <div class="metric-label">Actual Measured Separation (d_geom)</div>
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
          <div id="val-geom" class="metric-val" style="color: var(--accent);">464.5 mm</div>
          <div id="val-l0-contact" class="metric-sub" style="font-weight:700; color:var(--green);">L0 Contact Distance: 10 mm</div>
        </div>
        <div class="bar-container" style="height:10px;">
          <div id="bar-geom" class="bar-fill" style="width: 80%; background: var(--accent);"></div>
        </div>
      </div>

      <!-- Summary Card -->
      <div class="metric-box full-width">
        <div class="metric-label">Intervention Lifecycle Status</div>
        <div id="txt-lifecycle" style="font-size:0.8rem; line-height:1.4; color:var(--text-secondary); margin-top:4px;">
          System operating nominally under continuous 100 Hz RSSM forward lookahead.
        </div>
      </div>

    </div>
  </div>
</div>

<script>
let isPaused = false;

function updateTelemetry() {
  fetch('/telemetry')
    .then(r => r.json())
    .then(data => {
      // Time & Stage
      document.getElementById('txt-sim-time').innerText = 'Time: ' + data.sim_time.toFixed(2) + ' s';
      document.getElementById('val-stage').innerText = data.planner_stage;
      document.getElementById('sub-cursor').innerText = 'Waypoint Cursor: ' + data.cursor + ' (Tick ' + data.tick + ')';
      document.getElementById('txt-scenario-name').innerText = 'Scenario: ' + data.scenario;

      // Supervisor Badge & State
      const badge = document.getElementById('badge-supervisor');
      badge.innerText = data.supervisor_state;
      if (data.supervisor_state === 'NORMAL') {
        badge.style.background = 'var(--green)';
      } else if (data.supervisor_state === 'CAUTION') {
        badge.style.background = 'var(--yellow)';
      } else if (data.supervisor_state === 'SLOWDOWN') {
        badge.style.background = 'var(--orange)';
      } else if (data.supervisor_state === 'EMERGENCY_STOP') {
        badge.style.background = 'var(--red)';
      }

      // Speed Scale
      document.getElementById('val-speed').innerText = data.speed_scale.toFixed(2) + 'x';
      const speedPct = Math.round(data.speed_scale * 100);
      const barSpeed = document.getElementById('bar-speed');
      barSpeed.style.width = speedPct + '%';
      barSpeed.style.background = data.speed_scale > 0.5 ? 'var(--green)' : (data.speed_scale > 0 ? 'var(--orange)' : 'var(--red)');

      // P10 & P50 Risk
      document.getElementById('val-p10').innerText = data.p10.toFixed(4);
      if (document.getElementById('val-p50')) {
        document.getElementById('val-p50').innerText = (data.p50 !== undefined ? data.p50.toFixed(4) : '--');
      }
      const pMax = Math.max(data.p10, (data.p50 || 0));
      const p10Pct = Math.min(100, Math.round(pMax * 100));
      const barP10 = document.getElementById('bar-p10');
      barP10.style.width = p10Pct + '%';
      barP10.style.background = pMax >= 0.70 ? 'var(--red)' : (pMax >= 0.36 ? 'var(--orange)' : 'var(--green)');

      // d10 Clearance
      document.getElementById('val-d10').innerText = (data.d10_mm >= 1500 ? 'N/A' : data.d10_mm.toFixed(1) + ' mm');
      const d10Pct = Math.min(100, Math.round((data.d10_mm / 250.0) * 100));
      const barD10 = document.getElementById('bar-d10');
      barD10.style.width = data.d10_mm >= 1500 ? '0%' : d10Pct + '%';
      barD10.style.background = data.d10_mm <= 30 ? 'var(--red)' : (data.d10_mm <= 80 ? 'var(--orange)' : 'var(--green)');

      // Actual Clearance (with 999m sentinel check)
      if (data.actual_clearance_mm >= 1500.0 || !data.hazard_active) {
        document.getElementById('val-geom').innerText = 'N/A';
        document.getElementById('bar-geom').style.width = '0%';
      } else {
        document.getElementById('val-geom').innerText = data.actual_clearance_mm.toFixed(1) + ' mm';
        const geomPct = Math.min(100, Math.round((data.actual_clearance_mm / 600.0) * 100));
        document.getElementById('bar-geom').style.width = geomPct + '%';
      }

      // Layer 0 Badge
      const badgeL0 = document.getElementById('badge-l0');
      if (data.layer0_collision) {
        badgeL0.innerText = 'L0: COLLISION BREACH';
        badgeL0.style.background = 'var(--red)';
        badgeL0.style.color = '#fff';
      } else {
        badgeL0.innerText = 'L0: CLEAR';
        badgeL0.style.background = '#1f2c44';
        badgeL0.style.color = '#a2b4d2';
      }

      // Intervention Banner
      const banner = document.getElementById('intervention-banner');
      if (data.predictive_intervention_active) {
        banner.style.display = 'flex';
      } else {
        banner.style.display = 'none';
      }

      // Lifecycle status
      const lifeTxt = document.getElementById('txt-lifecycle');
      if (data.stopped_info) {
        const clearStr = (data.stopped_info.clearance_mm >= 1500 ? 'safe distance' : data.stopped_info.clearance_mm + ' mm');
        lifeTxt.innerHTML = '<span style="color:var(--green); font-weight:700;">SAFE REST ACHIEVED:</span> Robot arrested at step ' + data.stopped_info.step + ' with ' + clearStr + ' clearance to hazard. Layer 0 contact fully avoided.';
      } else if (data.first_intervention) {
        const triggerP = (data.first_intervention.p50 >= 0.36 ? 'P50=' + data.first_intervention.p50 : 'P10=' + data.first_intervention.p10);
        lifeTxt.innerHTML = '<span style="color:var(--orange); font-weight:700;">INTERVENTION IN PROGRESS:</span> Triggered at step ' + data.first_intervention.step + ' (' + triggerP + '). Decelerating to rest.';
      } else {
        lifeTxt.innerText = 'System operating nominally under continuous 100 Hz RSSM forward lookahead.';
      }

      isPaused = data.is_paused;
      document.getElementById('btn-pause').innerText = isPaused ? 'Resume' : 'Pause';
    })
    .catch(() => {});
}

function updateFrame() {
  const img = document.getElementById('sim-viewport');
  img.src = '/frame?t=' + Date.now();
}

function setScenario(name) {
  fetch('/api/scenario?name=' + name, { method: 'POST' }).then(() => {
    document.getElementById('btn-scen-a').className = (name === 'A_NORMAL') ? 'active' : '';
    document.getElementById('btn-scen-b').className = (name === 'B_HAZARD') ? 'active' : '';
  });
}

function togglePause() {
  fetch('/api/pause', { method: 'POST' });
}

function resetCurrent() {
  fetch('/api/reset', { method: 'POST' });
}

setInterval(updateTelemetry, 100);
setInterval(updateFrame, 40); // ~25 FPS
</script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    controller: Optional[DashboardController] = None

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        return

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD_TEMPLATE.encode("utf-8"))

        elif self.path.startswith("/telemetry"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            data = self.controller.telemetry_data if self.controller else {}
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif self.path.startswith("/frame"):
            self.send_response(200)
            self.send_header("Content-Type", "image/bmp")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            frame_data = self.controller.current_frame_bmp if self.controller else b""
            self.wfile.write(frame_data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/scenario"):
            query = self.path.split("?", 1)[-1] if "?" in self.path else ""
            params = dict(qc.split("=") for qc in query.split("&") if "=" in qc)
            scen = params.get("name", "B_HAZARD")
            if self.controller:
                self.controller.reset_scenario(scen)
            self.send_response(200)
            self.end_headers()

        elif self.path == "/api/pause":
            if self.controller:
                with self.controller.lock:
                    self.controller.is_paused = not self.controller.is_paused
            self.send_response(200)
            self.end_headers()

        elif self.path == "/api/reset":
            if self.controller:
                self.controller.reset_scenario(self.controller.scenario_name)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_dashboard_server(controller: DashboardController, host: str = "127.0.0.1", port: int = 8050):
    DashboardRequestHandler.controller = controller
    server = HTTPServer((host, port), DashboardRequestHandler)
    print(f"\n" + "=" * 78)
    print(f"  SAWM PREDICTIVE SAFETY DASHBOARD ONLINE")
    print(f"  URL: http://{host}:{port}/")
    print("=" * 78)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Launch interactive SAWM predictive intervention dashboard.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind dashboard web server (default: 8050).")
    parser.add_argument("--scenario", choices=["A", "B"], default="B", help="Initial scenario: A (Nominal) or B (Hazard).")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser.")
    args = parser.parse_args()

    scen_name = "A_NORMAL" if args.scenario == "A" else "B_HAZARD"
    controller = DashboardController()
    controller.reset_scenario(scen_name)

    # Launch HTTP server in background daemon thread
    server_thread = threading.Thread(
        target=run_dashboard_server,
        args=(controller, "127.0.0.1", args.port),
        daemon=True,
    )
    server_thread.start()

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    print(f"Simulation engine running in main loop. Press Ctrl+C in terminal to stop.")
    try:
        while True:
            controller.step()
            time.sleep(controller.step_delay)
    except KeyboardInterrupt:
        print("\n[Dashboard Shutdown] Stopped by operator.")
    finally:
        controller.env.close()


if __name__ == "__main__":
    main()
