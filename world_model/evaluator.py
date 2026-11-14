"""
Multi-Horizon Latent Rollout Evaluator for SAWM RSSM.

Features:
- Open-loop multi-step latent imagination at H in {1, 5, 10, 20, 50}
- Denormalization to physical units:
  - Robot joints (rad)
  - EE position (mm)
  - EE orientation (geodesic error in degrees)
  - Object position (mm)
  - Object orientation (degrees)
  - Safety clearance d_min (mm)
  - Contact bit accuracy (%)
- Evaluates separately across validation subsets (nominal, static, moving proxy)
- Latency profiling on GPU/CPU
"""

import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

from world_model.rssm import RSSM
from world_model.dataset import TrajectoryNormalizer


class MultiHorizonEvaluator:
    """Evaluates open-loop imagination accuracy and latency across multiple horizons."""

    def __init__(self, model: RSSM, normalizer: TrajectoryNormalizer, device: torch.device):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = device

    @staticmethod
    def geodesic_angle_degrees(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Compute geodesic angular error in degrees between pairs of unit quaternions."""
        inner = np.abs(np.sum(q1 * q2, axis=-1))
        clamped = np.clip(inner, 0.0, 1.0 - 1e-7)
        return 2.0 * np.arccos(clamped) * (180.0 / np.pi)

    def evaluate_episode(self, obs_raw: np.ndarray, act: np.ndarray, horizons: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Evaluate multi-horizon open-loop rollouts starting at multiple rollout points in an episode.
        
        Args:
            obs_raw: (T, 49) Ground-truth raw physical observations
            act: (T-1, 8) Ground-truth actions
            horizons: List of horizons, e.g. [1, 5, 10, 20, 50]
            
        Returns:
            Dict mapping horizon H -> dict of physical metrics
        """
        T = len(obs_raw)
        max_H = max(horizons)

        # Normalize observation sequence for model input
        obs_norm = self.normalizer.normalize(obs_raw) if self.normalizer.is_fit else obs_raw

        # We evaluate rollouts starting at steps t = 10, 60, 110, 160, 210, etc.
        start_indices = [t for t in range(5, T - max_H - 1, 30)]
        if not start_indices:
            start_indices = [0]

        results = {H: {
            "norm_rmse": [],
            "joint_err_rad": [],
            "ee_pos_err_mm": [],
            "ee_rot_err_deg": [],
            "obj_pos_err_mm": [],
            "obj_rot_err_deg": [],
            "clearance_err_mm": [],
            "contact_acc": [],
        } for H in horizons}

        with torch.no_grad():
            for t_start in start_indices:
                # 1. Initialize latent state (filter from t=0 to t_start)
                context_obs = torch.tensor(obs_norm[max(0, t_start - 5) : t_start + 1], dtype=torch.float32, device=self.device).unsqueeze(0)
                context_act = torch.tensor(act[max(0, t_start - 5) : t_start], dtype=torch.float32, device=self.device).unsqueeze(0)

                forward_out = self.model(context_obs, context_act)
                h_0 = forward_out["deterministic_states"][:, -1]
                s_0 = forward_out["stochastic_states"][:, -1]

                # 2. Open-loop imagine rollout for max_H steps
                act_future = torch.tensor(act[t_start : t_start + max_H], dtype=torch.float32, device=self.device).unsqueeze(0)
                imagined_norm, _, _ = self.model.imagine(h_0, s_0, act_future)

                imagined_np = imagined_norm.squeeze(0).cpu().numpy()  # (max_H, 49)
                imagined_raw = self.normalizer.denormalize(imagined_np) if self.normalizer.is_fit else imagined_np

                # Re-normalize quaternions on denormalized predictions
                for tau in range(max_H):
                    q_ee = imagined_raw[tau, 18:22]
                    imagined_raw[tau, 18:22] = q_ee / max(1e-8, np.linalg.norm(q_ee))
                    q_obj = imagined_raw[tau, 25:29]
                    imagined_raw[tau, 25:29] = q_obj / max(1e-8, np.linalg.norm(q_obj))

                gt_raw = obs_raw[t_start + 1 : t_start + 1 + max_H]
                gt_norm = obs_norm[t_start + 1 : t_start + 1 + max_H]

                # Evaluate metrics at each horizon H
                for H in horizons:
                    pred_H = imagined_raw[H - 1]
                    true_H = gt_raw[H - 1]

                    pred_norm_H = imagined_np[H - 1]
                    true_norm_H = gt_norm[H - 1]

                    # 1. Overall normalized RMSE
                    norm_rmse = float(np.sqrt(np.mean((pred_norm_H - true_norm_H) ** 2)))
                    results[H]["norm_rmse"].append(norm_rmse)

                    # 2. Joint angles (rad)
                    j_err = float(np.sqrt(np.mean((pred_H[0:7] - true_H[0:7]) ** 2)))
                    results[H]["joint_err_rad"].append(j_err)

                    # 3. EE Position (mm)
                    ee_err = float(np.linalg.norm(pred_H[15:18] - true_H[15:18]) * 1000.0)
                    results[H]["ee_pos_err_mm"].append(ee_err)

                    # 4. EE Orientation (deg)
                    ee_rot = float(self.geodesic_angle_degrees(true_H[18:22], pred_H[18:22]))
                    results[H]["ee_rot_err_deg"].append(ee_rot)

                    # 5. Object Position (mm)
                    obj_err = float(np.linalg.norm(pred_H[22:25] - true_H[22:25]) * 1000.0)
                    results[H]["obj_pos_err_mm"].append(obj_err)

                    # 6. Object Orientation (deg)
                    obj_rot = float(self.geodesic_angle_degrees(true_H[25:29], pred_H[25:29]))
                    results[H]["obj_rot_err_deg"].append(obj_rot)

                    # 7. Hazard Clearance (mm)
                    d_err = float(abs(pred_H[45] - true_H[45]) * 1000.0)
                    results[H]["clearance_err_mm"].append(d_err)

                    # 8. Contact Bit Accuracy (%)
                    c_pred = (pred_H[47:49] > 0.5).astype(int)
                    c_true = (true_H[47:49] > 0.5).astype(int)
                    c_acc = float(np.mean(c_pred == c_true) * 100.0)
                    results[H]["contact_acc"].append(c_acc)

        # Average across rollout evaluation points
        summary = {}
        for H in horizons:
            summary[H] = {metric: float(np.mean(vals)) for metric, vals in results[H].items()}
        return summary

    def benchmark_inference_latency(self, horizons: List[int], num_trials: int = 50) -> Dict[int, float]:
        """Benchmark average GPU latency per imagination rollout in milliseconds."""
        latencies = {}
        B = 1
        with torch.no_grad():
            h_0 = torch.zeros(B, self.model.cfg.deterministic_dim, device=self.device)
            s_0 = torch.zeros(B, self.model.cfg.stochastic_dim, device=self.device)

            for H in horizons:
                act = torch.zeros(B, H, self.model.cfg.action_dim, device=self.device)
                # Warmup
                for _ in range(5):
                    self.model.imagine(h_0, s_0, act)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(num_trials):
                    self.model.imagine(h_0, s_0, act)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                t1 = time.perf_counter()

                avg_ms = ((t1 - t0) / num_trials) * 1000.0
                latencies[H] = float(avg_ms)
        return latencies
