"""
SAWM - Milestone 4 Multi-Horizon Model Evaluator Script.
Runs comprehensive open-loop latent imagination evaluation on held-out validation trajectories.

Outputs:
- Multi-horizon prediction error table (H = 1, 5, 10, 20, 50)
- Physical unit breakdowns (mm, rad, deg, %)
- Validation subset breakdown (Nominal, Static Obstacle, Moving Proxy)
- GPU inference latency benchmark
- Saves evaluation summary to `results/milestone4_evaluation.json`

Usage:
  python scripts/evaluate_world_model.py
"""

import os
import sys
import glob
import json
import yaml
import argparse
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import TrajectoryNormalizer
from world_model.evaluator import MultiHorizonEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 4 - Multi-Horizon Evaluator")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rssm_best.pt", help="Path to checkpoint")
    parser.add_argument("--normalizer", type=str, default="checkpoints/normalizer.pt", help="Path to normalizer")
    parser.add_argument("--val-dir", type=str, default="data/trajectories/val", help="Path to validation directory")
    parser.add_argument("--output-json", type=str, default="results/milestone4_evaluation.json", help="Path to output JSON")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 80)
    print("SAWM - Milestone 4: Multi-Horizon World Model Evaluation")
    print("=" * 80)
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Normalizer:  {args.normalizer}")
    print(f"Val Dir:     {args.val_dir}")
    print("=" * 80)

    # 1. Load Checkpoint and Rebuild Model
    assert os.path.exists(args.checkpoint), f"Checkpoint not found at: {args.checkpoint}"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    m_cfg = cfg.get("model", {})
    rssm_cfg = RSSMConfig(
        obs_dim=m_cfg.get("obs_dim", 49),
        action_dim=m_cfg.get("action_dim", 8),
        encoder_hidden_dim=m_cfg.get("encoder_hidden_dim", 128),
        deterministic_dim=m_cfg.get("deterministic_dim", 128),
        stochastic_dim=m_cfg.get("stochastic_dim", 32),
        prior_hidden_dim=m_cfg.get("prior_hidden_dim", 128),
        posterior_hidden_dim=m_cfg.get("posterior_hidden_dim", 128),
        decoder_hidden_dim=m_cfg.get("decoder_hidden_dim", 128),
        min_std=m_cfg.get("min_std", 0.1),
        max_std=m_cfg.get("max_std", 10.0),
    )
    model = RSSM(rssm_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded RSSM model checkpoint from epoch {ckpt.get('epoch', 'N/A')} (Val Loss: {ckpt.get('val_loss', 'N/A'):.4f})")

    # 2. Load Normalizer
    normalizer = TrajectoryNormalizer(obs_dim=rssm_cfg.obs_dim)
    if os.path.exists(args.normalizer):
        normalizer.load(args.normalizer)
        print(f"[INFO] Loaded normalizer from {args.normalizer}")

    # 3. Locate Validation Files & Subsets
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.npz")))
    assert len(val_files) > 0, f"No validation files found in {args.val_dir}"
    print(f"[INFO] Found {len(val_files)} held-out validation episodes.")

    evaluator = MultiHorizonEvaluator(model, normalizer, device)
    horizons = cfg.get("evaluation", {}).get("horizons", [1, 5, 10, 20, 50])

    # 4. Evaluate Across All Validation Episodes
    overall_results = {H: {m: [] for m in [
        "norm_rmse", "joint_err_rad", "ee_pos_err_mm", "ee_rot_err_deg",
        "obj_pos_err_mm", "obj_rot_err_deg", "clearance_err_mm", "contact_acc"
    ]} for H in horizons}

    subset_results: Dict[str, Dict[int, Dict[str, List[float]]]] = {}

    for fpath in val_files:
        data = np.load(fpath, allow_pickle=True)
        obs_raw = data["observations"]
        act = data["actions"]
        meta = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
        subset = meta.get("subset", "nominal")

        if subset not in subset_results:
            subset_results[subset] = {H: {m: [] for m in overall_results[H]} for H in horizons}

        ep_summary = evaluator.evaluate_episode(obs_raw, act, horizons)

        for H in horizons:
            for metric, val in ep_summary[H].items():
                overall_results[H][metric].append(val)
                subset_results[subset][H][metric].append(val)

    # 5. Average Results
    overall_avg = {
        H: {m: float(np.mean(vals)) for m, vals in overall_results[H].items()}
        for H in horizons
    }

    subset_avg = {
        subset: {
            H: {m: float(np.mean(vals)) for m, vals in subset_results[subset][H].items()}
            for H in horizons
        }
        for subset in subset_results
    }

    # 6. Benchmark Inference Latency
    print("\n[INFO] Benchmarking latent imagination rollout inference latency...")
    latencies = evaluator.benchmark_inference_latency(horizons, num_trials=100)

    # 7. Print Formatted Tables
    print("\n" + "=" * 95)
    print("TABLE 1: MULTI-HORIZON OPEN-LOOP PREDICTION PERFORMANCE (HELD-OUT VALIDATION)")
    print("=" * 95)
    print(f"{'Horizon H':<10} | {'Time (s)':<9} | {'Norm RMSE':<10} | {'Joints (rad)':<12} | {'EE Pos (mm)':<12} | {'EE Rot (deg)':<12} | {'Obj Pos (mm)':<12} | {'Clearance (mm)':<14}")
    print("-" * 95)
    for H in horizons:
        dt_s = H * 0.01
        m = overall_avg[H]
        print(
            f"{H:<10d} | "
            f"{dt_s:<9.2f} | "
            f"{m['norm_rmse']:<10.4f} | "
            f"{m['joint_err_rad']:<12.4f} | "
            f"{m['ee_pos_err_mm']:<12.2f} | "
            f"{m['ee_rot_err_deg']:<12.2f} | "
            f"{m['obj_pos_err_mm']:<12.2f} | "
            f"{m['clearance_err_mm']:<14.2f}"
        )
    print("=" * 95)

    print("\n" + "=" * 80)
    print("TABLE 2: VALIDATION SUBSET BREAKDOWN AT HORIZON H = 10 (0.10s Ahead)")
    print("=" * 80)
    print(f"{'Subset':<18} | {'Episodes':<9} | {'Norm RMSE':<10} | {'EE Pos (mm)':<12} | {'Obj Pos (mm)':<12} | {'Clearance (mm)':<14}")
    print("-" * 80)
    for subset, sub_dict in subset_avg.items():
        count = len(subset_results[subset][10]["norm_rmse"])
        m = sub_dict[10]
        print(
            f"{subset:<18s} | "
            f"{count:<9d} | "
            f"{m['norm_rmse']:<10.4f} | "
            f"{m['ee_pos_err_mm']:<12.2f} | "
            f"{m['obj_pos_err_mm']:<12.2f} | "
            f"{m['clearance_err_mm']:<14.2f}"
        )
    print("=" * 80)

    print("\n" + "=" * 55)
    print("TABLE 3: MODEL INFERENCE LATENCY BENCHMARK")
    print("=" * 55)
    print(f"{'Horizon H':<12} | {'Sim Time (s)':<14} | {'GPU Latency (ms)':<16}")
    print("-" * 55)
    for H in horizons:
        print(f"{H:<12d} | {H * 0.01:<14.2f} | {latencies[H]:<16.2f}")
    print("=" * 55)

    # 8. Save Evaluation JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    out_payload = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", None),
        "val_loss": ckpt.get("val_loss", None),
        "device": str(device),
        "overall_by_horizon": overall_avg,
        "subsets_by_horizon": subset_avg,
        "latency_ms_by_horizon": latencies,
    }
    with open(args.output_json, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\n[INFO] Complete evaluation results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
