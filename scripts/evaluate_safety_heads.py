"""
Comprehensive Evaluation Pipeline for Milestone 5 Safety Heads.

Evaluates exclusively on the held-out, untouched Test Partition (17 episodes):
- 10 M4 clean baseline episodes
- 7 augmented collision / near-miss episodes across all 5 hazard categories

Reports:
1. Transition-Level Metrics across H in {1, 5, 10, 20, 50}:
   - Collision Recall, Precision, F1
   - False Positive Rate (FPR) on clean baseline episodes
   - False Alarms on hard near-miss episodes
   - Brier calibration score
   - Clearance MAE (overall and within danger zone d < 0.08m)
2. Category-Specific Breakdown across all 5 hazard categories + clean baseline.
3. Episode-Level Metrics:
   - Early warning lead time relative to first hazardous contact
   - Early warning success rate (>= 0.10s lead time) with Wilson score 95% confidence intervals
   - Episode-level false alarm rate on non-colliding episodes
4. Actual Inference Latency Benchmarks (GPU) at each horizon.

Strictly distinguishes frame-level and episode-level metrics.
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model.rssm import RSSM
from world_model.dataset import TrajectoryNormalizer
from safety.predictive_physical_head import PhysicalRiskHead, HORIZONS, HORIZON_INDICES
from safety.predictive_contextual_head import PredictiveContextualHarmHead
from safety.predictive_safety_system import PredictiveSafetySystem


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate Wilson score 95% confidence interval for binomial proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    z = 1.95996  # 95% confidence
    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    spread = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4 * (n ** 2))))
    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))
    return float(p_hat), lower, upper


def run_evaluation():
    print("=" * 80)
    print("SAWM - Milestone 5 Safety Heads Comprehensive Evaluation")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Manifest
    manifest_path = "data/dataset_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    test_eps = manifest["partitions"]["test"]["episodes"]
    print(f"\nEvaluating ONCE on untouched Test Partition: {len(test_eps)} episodes.")
    assert len(test_eps) == 17

    # 2. Load Checkpoints
    normalizer = TrajectoryNormalizer()
    normalizer.load("checkpoints/normalizer.pt")

    rssm = RSSM().to(device)
    rssm_ckpt = torch.load("checkpoints/rssm_best.pt", map_location=device, weights_only=False)
    rssm.load_state_dict(rssm_ckpt["model_state_dict"])
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad = False

    phys_ckpt = torch.load("checkpoints/physical_head_best.pt", map_location=device, weights_only=False)
    phys_head = PhysicalRiskHead(latent_dim=160).to(device)
    phys_head.load_state_dict(phys_ckpt["model_state_dict"])
    phys_head.eval()

    with open("checkpoints/safety_calibrator.json", "r") as f:
        cal_meta = json.load(f)

    temp_cal = float(cal_meta["temperature"])
    tau_coll = float(cal_meta["collision_threshold"])
    print(f"Loaded PhysicalRiskHead (31,882 params) | Calibrated T: {temp_cal:.4f} | tau_coll: {tau_coll:.2f}")

    ctx_head = PredictiveContextualHarmHead()
    safety_system = PredictiveSafetySystem(
        physical_head=phys_head,
        contextual_head=ctx_head,
        collision_threshold=tau_coll,
    )

    # -------------------------------------------------------------------------
    # 3. Transition-Level & Episode-Level Evaluation Loop
    # -------------------------------------------------------------------------
    # Transition metric accumulators per horizon
    horizon_stats = {
        h: {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "fp_clean": 0, "tn_clean": 0,
            "fp_near": 0, "tn_near": 0,
            "brier_sum": 0.0, "brier_count": 0,
            "clear_err_sum": 0.0, "clear_count": 0,
            "danger_clear_err_sum": 0.0, "danger_clear_count": 0,
        }
        for h in HORIZONS
    }

    # Category-specific tracking for H=10
    categories = [
        "m4_baseline_mixed",
        "pick_approach_collision",
        "transit_corridor_collision",
        "place_descent_collision",
        "dynamic_proxy_collision",
        "boundary_grazing_hard_negatives",
    ]
    category_stats_h10 = {
        cat: {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "brier_sum": 0.0, "brier_count": 0,
            "clear_err_sum": 0.0, "clear_count": 0,
            "danger_clear_err_sum": 0.0, "danger_clear_count": 0,
            "episodes": 0,
        }
        for cat in categories
    }

    # Episode-level early warning tracking
    episode_results = []
    total_collision_episodes = 0
    successful_early_warnings = 0
    late_warnings = 0
    missed_collisions = 0
    nominal_false_alarms = 0
    total_nominal_episodes = 0

    print("\nRunning test evaluation across 17 test episodes...")

    for ep_idx, ep in enumerate(test_eps):
        fpath = ep["filepath"]
        cat = ep.get("category", "m4_baseline")
        data = np.load(fpath, allow_pickle=True)
        obs_raw = data["observations"]
        act_raw = data["actions"]
        T = len(obs_raw)

        if "hazard_contacts" in data:
            c_haz = data["hazard_contacts"]
        else:
            c_haz = obs_raw[:, 48]

        if "clearances" in data:
            clear_raw = data["clearances"]
        else:
            clear_raw = obs_raw[:, 45]

        has_collision = (np.sum(c_haz) > 0)
        norm_obs = normalizer.normalize(obs_raw)

        category_stats_h10[cat]["episodes"] += 1

        # Determine first contact step if collision occurred
        if has_collision:
            total_collision_episodes += 1
            t_contact_star = int(np.where(c_haz > 0.5)[0][0])
        else:
            total_nominal_episodes += 1
            t_contact_star = None

        earliest_alarm_step_h10 = None

        # Slices along trajectory: evaluate every step from t=49 to T-2
        for t in range(49, T - 1):
            w_obs = norm_obs[t - 49 : t + 1]
            w_act = act_raw[t - 49 : t]

            rem_act = act_raw[t : min(t + 50, len(act_raw))]
            n_rem = len(rem_act)
            if n_rem < 50:
                pad_act = np.tile(act_raw[-1], (50 - n_rem, 1))
                f_act = np.vstack([rem_act, pad_act]) if n_rem > 0 else np.tile(act_raw[-1], (50, 1))
            else:
                f_act = rem_act[:50]

            # Run through frozen RSSM
            with torch.no_grad():
                w_obs_t = torch.tensor(w_obs, dtype=torch.float32, device=device).unsqueeze(0)
                w_act_t = torch.tensor(w_act, dtype=torch.float32, device=device).unsqueeze(0)
                f_act_t = torch.tensor(f_act, dtype=torch.float32, device=device).unsqueeze(0)

                warm_out = rssm(w_obs_t, w_act_t)
                h0 = warm_out["deterministic_states"][:, -1]
                s0 = warm_out["stochastic_states"][:, -1]
                img_obs, img_h, img_s = rssm.imagine(h0, s0, f_act_t)

                z_rollout = torch.cat([img_h, img_s], dim=-1)
                z_horizons = z_rollout[:, HORIZON_INDICES, :]

                # Run PhysicalRiskHead
                phys_out = phys_head(z_horizons, use_temperature=True)
                probs = phys_out["calibrated_probs"][0].cpu().numpy()
                clearances = phys_out["clearances"][0].cpu().numpy()

            # Record metrics for each horizon
            for h_i, h in enumerate(HORIZONS):
                if t + h <= T - 1:
                    # Ground truth labels
                    y_true = float(np.max(c_haz[t + 1 : t + h + 1]))
                    d_true = max(0.0, float(np.min(clear_raw[t + 1 : t + h + 1])))

                    p_pred = float(probs[h_i])
                    d_pred = float(clearances[h_i])
                    pred_binary = 1.0 if (p_pred >= tau_coll) else 0.0

                    hs = horizon_stats[h]

                    if pred_binary == 1.0 and y_true == 1.0:
                        hs["tp"] += 1
                    elif pred_binary == 1.0 and y_true == 0.0:
                        hs["fp"] += 1
                    elif pred_binary == 0.0 and y_true == 0.0:
                        hs["tn"] += 1
                    else:
                        hs["fn"] += 1

                    # Clean baseline vs near-miss false positive separation
                    if y_true == 0.0:
                        if cat in ["m4_baseline", "m4_baseline_mixed"]:
                            if pred_binary == 1.0:
                                hs["fp_clean"] += 1
                            else:
                                hs["tn_clean"] += 1
                        elif cat == "boundary_grazing_hard_negatives":
                            if pred_binary == 1.0:
                                hs["fp_near"] += 1
                            else:
                                hs["tn_near"] += 1

                    # Brier score
                    hs["brier_sum"] += (p_pred - y_true) ** 2
                    hs["brier_count"] += 1

                    # Clearance MAE
                    clear_err = abs(d_pred - d_true)
                    hs["clear_err_sum"] += clear_err
                    hs["clear_count"] += 1

                    if d_true < 0.08:
                        hs["danger_clear_err_sum"] += clear_err
                        hs["danger_clear_count"] += 1

                    # Track H=10 category stats
                    if h == 10:
                        cs = category_stats_h10[cat]
                        if pred_binary == 1.0 and y_true == 1.0:
                            cs["tp"] += 1
                        elif pred_binary == 1.0 and y_true == 0.0:
                            cs["fp"] += 1
                        elif pred_binary == 0.0 and y_true == 0.0:
                            cs["tn"] += 1
                        else:
                            cs["fn"] += 1

                        cs["brier_sum"] += (p_pred - y_true) ** 2
                        cs["brier_count"] += 1
                        cs["clear_err_sum"] += clear_err
                        cs["clear_count"] += 1
                        if d_true < 0.08:
                            cs["danger_clear_err_sum"] += clear_err
                            cs["danger_clear_count"] += 1

            # Check for earliest alarm at H=10
            p_h10 = float(probs[2])
            d_h10 = float(clearances[2])
            if (p_h10 >= tau_coll or d_h10 <= 0.08) and earliest_alarm_step_h10 is None:
                earliest_alarm_step_h10 = t

        # Evaluate episode-level outcome
        if has_collision:
            if earliest_alarm_step_h10 is not None:
                lead_time_steps = t_contact_star - earliest_alarm_step_h10
                lead_time_sec = lead_time_steps * 0.01

                if lead_time_steps >= 10:
                    early_status = "SUCCESSFUL_EARLY_WARNING"
                    successful_early_warnings += 1
                elif lead_time_steps > 0:
                    early_status = "LATE_WARNING"
                    late_warnings += 1
                else:
                    early_status = "MISSED_COLLISION"
                    missed_collisions += 1
            else:
                lead_time_steps = -999
                lead_time_sec = -9.99
                early_status = "MISSED_COLLISION"
                missed_collisions += 1

            ep_id = ep.get("episode_id", ep.get("global_episode_id", ep_idx))
            episode_results.append({
                "episode_id": ep_id,
                "category": cat,
                "has_collision": True,
                "contact_step": t_contact_star,
                "alarm_step": earliest_alarm_step_h10,
                "lead_time_steps": lead_time_steps,
                "lead_time_seconds": lead_time_sec,
                "status": early_status,
            })
        else:
            has_false_alarm = (earliest_alarm_step_h10 is not None)
            if has_false_alarm:
                nominal_false_alarms += 1
                nom_status = "FALSE_ALARM"
            else:
                nom_status = "CLEAN_SAFE"

            ep_id = ep.get("episode_id", ep.get("global_episode_id", ep_idx))
            episode_results.append({
                "episode_id": ep_id,
                "category": cat,
                "has_collision": False,
                "alarm_step": earliest_alarm_step_h10,
                "status": nom_status,
            })

    # -------------------------------------------------------------------------
    # 4. Latency Benchmarking (100 runs per horizon on GPU)
    # -------------------------------------------------------------------------
    print("\n--- Latency Benchmarking on GPU (100 Iterations per Horizon) ---")
    latency_results = {}
    dummy_obs_warm = torch.randn(1, 50, 49, device=device)
    dummy_act_warm = torch.randn(1, 49, 8, device=device)

    for h in HORIZONS:
        dummy_f_act = torch.randn(1, h, 8, device=device)

        # Warmup GPU
        for _ in range(10):
            with torch.no_grad():
                w_out = rssm(dummy_obs_warm, dummy_act_warm)
                h0, s0 = w_out["deterministic_states"][:, -1], w_out["stochastic_states"][:, -1]
                _, img_h, img_s = rssm.imagine(h0, s0, dummy_f_act)
                z = torch.cat([img_h, img_s], dim=-1)
                _ = phys_head(z)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N_bench = 100
        for _ in range(N_bench):
            with torch.no_grad():
                w_out = rssm(dummy_obs_warm, dummy_act_warm)
                h0, s0 = w_out["deterministic_states"][:, -1], w_out["stochastic_states"][:, -1]
                _, img_h, img_s = rssm.imagine(h0, s0, dummy_f_act)
                z = torch.cat([img_h, img_s], dim=-1)
                _ = phys_head(z)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_lat_ms = (t1 - t0) / N_bench * 1000
        latency_results[h] = avg_lat_ms
        print(f"  Horizon H={h:2d} ({h*0.01:.2f}s lookahead): {avg_lat_ms:.2f} ms")

    # -------------------------------------------------------------------------
    # 5. Compile and Print Comprehensive Metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MILESTONE 5 EVALUATION RESULTS ON UNTOUCHED TEST SPLIT (17 EPISODES)")
    print("=" * 80)

    # 1. Multi-Horizon Transition-Level Table
    print("\n--- 1. Multi-Horizon Transition-Level Metrics ---")
    print("Horizon | Time (s) | Recall   | Precision | F1 Score | Brier  | Clearance MAE | Danger Zone MAE | Clean FPR | Near-Miss FA")
    print("-----------------------------------------------------------------------------------------------------------------------")

    final_horizon_metrics = {}
    for h in HORIZONS:
        s = horizon_stats[h]
        tp, fp, tn, fn = s["tp"], s["fp"], s["tn"], s["fn"]
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        brier = s["brier_sum"] / max(1, s["brier_count"])
        clear_mae = (s["clear_err_sum"] / max(1, s["clear_count"])) * 1000
        danger_mae = (s["danger_clear_err_sum"] / max(1, s["danger_clear_count"])) * 1000 if s["danger_clear_count"] > 0 else 0.0

        fpr_clean = (s["fp_clean"] / max(1, s["fp_clean"] + s["tn_clean"])) * 100
        fa_near = (s["fp_near"] / max(1, s["fp_near"] + s["tn_near"])) * 100

        print(f"H={h:2d}    | {h*0.01:.2f}s    | {rec*100:6.1f}%  | {prec*100:7.1f}%  | {f1:8.4f} | {brier:6.4f} | {clear_mae:9.1f}mm | {danger_mae:11.1f}mm | {fpr_clean:7.2f}%  | {fa_near:9.2f}%")

        final_horizon_metrics[h] = {
            "horizon": h,
            "lookahead_seconds": h * 0.01,
            "recall": float(rec),
            "precision": float(prec),
            "f1": float(f1),
            "brier_score": float(brier),
            "clearance_mae_mm": float(clear_mae),
            "danger_zone_mae_mm": float(danger_mae),
            "clean_fpr_percent": float(fpr_clean),
            "near_miss_fa_percent": float(fa_near),
            "latency_ms": float(latency_results[h]),
        }

    # 2. Category-Specific Breakdown (H=10)
    print("\n--- 2. Category-Specific Breakdown at Primary Horizon H=10 (0.10s Lookahead) ---")
    print("Scenario Category                 | Episodes | Recall   | Precision | F1 Score | Brier  | Clearance MAE | Danger Zone MAE")
    print("-------------------------------------------------------------------------------------------------------------------------")

    final_category_metrics = {}
    for cat in categories:
        cs = category_stats_h10[cat]
        tp, fp, tn, fn = cs["tp"], cs["fp"], cs["tn"], cs["fn"]
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 1.0  # If no positive frames, recall is N/A (100%)
        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        f1 = 2 * prec * rec / max(1e-6, prec + rec) if (tp > 0) else 0.0
        brier = cs["brier_sum"] / max(1, cs["brier_count"])
        clear_mae = (cs["clear_err_sum"] / max(1, cs["clear_count"])) * 1000
        danger_mae = (cs["danger_clear_err_sum"] / max(1, cs["danger_clear_count"])) * 1000 if cs["danger_clear_count"] > 0 else 0.0

        rec_str = f"{rec*100:6.1f}%" if (tp + fn) > 0 else "   N/A  "
        prec_str = f"{prec*100:6.1f}%" if (tp + fp) > 0 else "  100.0%"

        print(f"{cat:<34s}| {cs['episodes']:8d} | {rec_str}  | {prec_str}  | {f1:8.4f} | {brier:6.4f} | {clear_mae:9.1f}mm | {danger_mae:11.1f}mm")

        final_category_metrics[cat] = {
            "episodes": cs["episodes"],
            "recall": float(rec) if (tp + fn) > 0 else None,
            "precision": float(prec),
            "f1": float(f1),
            "brier_score": float(brier),
            "clearance_mae_mm": float(clear_mae),
            "danger_zone_mae_mm": float(danger_mae),
        }

    # 3. Episode-Level Metrics & Wilson Score Confidence Intervals
    print("\n--- 3. Episode-Level Collision Warning & Wilson Confidence Intervals ---")
    p_hat, ci_low, ci_high = wilson_score_interval(successful_early_warnings, total_collision_episodes)
    nom_fa_rate, nom_low, nom_high = wilson_score_interval(nominal_false_alarms, total_nominal_episodes)

    print(f"  Total Colliding Episodes in Test Split: {total_collision_episodes}")
    print(f"  Successful Early Warnings (>= 0.10s):  {successful_early_warnings}/{total_collision_episodes} ({p_hat*100:.1f}%)")
    print(f"  Wilson 95% Confidence Interval:         [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
    print(f"  Late Warnings (< 0.10s lead time):      {late_warnings}")
    print(f"  Missed Collisions:                      {missed_collisions}")
    print(f"  Total Non-Colliding Episodes (Clean+Near): {total_nominal_episodes}")
    print(f"  Episode-Level False Alarms:             {nominal_false_alarms}/{total_nominal_episodes} ({nom_fa_rate*100:.1f}%) [95% CI: {nom_low*100:.1f}%, {nom_high*100:.1f}%]")

    # Individual Colliding Episodes Detail
    print("\n  Colliding Episodes Early Warning Detail:")
    print("  Ep ID | Category                        | Contact Step | Alarm Step | Lead Time (steps / sec) | Status")
    print("  ---------------------------------------------------------------------------------------------------------")
    for res in episode_results:
        if res["has_collision"]:
            print(f"  {res['episode_id']:5d} | {res['category']:<32s}| {res['contact_step']:12d} | {res['alarm_step']:10d} | {res['lead_time_steps']:5d} steps ({res['lead_time_seconds']:+.2f}s)   | {res['status']}")

    # -------------------------------------------------------------------------
    # 6. Save Full Evaluation Report to JSON
    # -------------------------------------------------------------------------
    eval_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "calibrated_parameters": {
            "temperature": temp_cal,
            "collision_threshold": tau_coll,
        },
        "multi_horizon_metrics": final_horizon_metrics,
        "category_metrics_h10": final_category_metrics,
        "episode_level_results": {
            "total_colliding_episodes": total_collision_episodes,
            "successful_early_warnings": successful_early_warnings,
            "point_estimate": p_hat,
            "wilson_95_ci_lower": ci_low,
            "wilson_95_ci_upper": ci_high,
            "late_warnings": late_warnings,
            "missed_collisions": missed_collisions,
            "total_nominal_episodes": total_nominal_episodes,
            "nominal_false_alarms": nominal_false_alarms,
            "episode_details": episode_results,
        },
        "latencies_ms": latency_results,
    }

    report_path = "results/milestone5_evaluation_report.json"
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nFull evaluation report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
