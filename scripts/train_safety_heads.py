"""
Training and Calibration Pipeline for PhysicalRiskHead (Milestone 5 Requirement).

Features:
1. Frozen M4 RSSM (199,089 params) used for 50-step posterior warmup and 50-step latent imagination.
2. Extracts raw c_haz labels before observation normalization.
3. Multi-horizon targets (H in {1, 5, 10, 20, 50}) with explicit validity masks near episode boundaries.
4. Trained exclusively on the 106-episode Training partition.
5. Calibrates temperature T and decision threshold tau_coll exclusively on the 17-episode Calibration partition.
6. The Test partition is strictly untouched.
7. Saves checkpoints/physical_head_best.pt and checkpoints/safety_calibrator.json.
"""

import os
import sys
import json
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model.rssm import RSSM
from world_model.dataset import TrajectoryNormalizer
from safety.predictive_physical_head import PhysicalRiskHead, HORIZONS, HORIZON_INDICES


class RolloutSafetyDataset(Dataset):
    """
    Dataset of conditioning windows for training PhysicalRiskHead.
    Each item contains:
    - warmup_obs: (50, 49) normalized observations ending at s_t
    - warmup_act: (49, 8) raw actions ending at a_{t-1}
    - future_act: (50, 8) planned future actions a_{t:t+49}
    - targets_coll: (5,) cumulative collision binary flags
    - targets_clear: (5,) cumulative minimum clearances (meters)
    - validity_mask: (5,) binary mask (1 if horizon within episode, 0 otherwise)
    """

    def __init__(
        self,
        episodes_metadata: list,
        normalizer: TrajectoryNormalizer,
        stride_nominal: int = 4,
        stride_collision: int = 1,
    ):
        self.samples = []
        self._build_dataset(episodes_metadata, normalizer, stride_nominal, stride_collision)

    def _build_dataset(self, episodes_metadata, normalizer, stride_nominal, stride_collision):
        total_eps = len(episodes_metadata)
        print(f"Building RolloutSafetyDataset from {total_eps} episodes...")

        for ep in episodes_metadata:
            fpath = ep["filepath"]
            data = np.load(fpath, allow_pickle=True)
            obs_raw = data["observations"]          # (T, 49)
            act_raw = data["actions"]               # (T-1, 8)
            T = len(obs_raw)

            # Extract raw ground-truth hazard contacts and clearances BEFORE normalizing
            if "hazard_contacts" in data:
                c_haz = data["hazard_contacts"]     # (T,)
            else:
                c_haz = obs_raw[:, 48]

            if "clearances" in data:
                clear_raw = data["clearances"]      # (T,)
            else:
                clear_raw = obs_raw[:, 45]

            # Normalize observations for RSSM
            norm_obs = normalizer.normalize(obs_raw)

            # Determine stride: stride=1 for colliding episodes to capture every contact transition
            is_collision = (np.sum(c_haz) > 0)
            stride = stride_collision if is_collision else stride_nominal

            # Conditioning points t: requires at least 49 prior steps for 50-step warmup
            for t in range(49, T - 1, stride):
                # 1. Warmup window (50 obs ending at s_t, 49 actions ending at a_{t-1})
                w_obs = norm_obs[t - 49 : t + 1]       # (50, 49)
                w_act = act_raw[t - 49 : t]            # (49, 8)

                # 2. Future actions for 50-step imagination
                rem_act = act_raw[t : min(t + 50, len(act_raw))]
                n_rem = len(rem_act)
                if n_rem < 50:
                    # Pad remaining steps with holding action
                    pad_act = np.tile(act_raw[-1], (50 - n_rem, 1))
                    f_act = np.vstack([rem_act, pad_act]) if n_rem > 0 else np.tile(act_raw[-1], (50, 1))
                else:
                    f_act = rem_act[:50]                # (50, 8)

                # 3. Ground-truth multi-horizon targets with explicit validity masks
                t_coll = []
                t_clear = []
                v_mask = []

                for h in HORIZONS:
                    if t + h <= T - 1:
                        # Horizon within recorded episode
                        c_h = float(np.max(c_haz[t + 1 : t + h + 1]))
                        d_h = float(np.min(clear_raw[t + 1 : t + h + 1]))
                        # Clamped to non-negative meters (penetrations treated as 0.0 clearance)
                        d_h_clamped = max(0.0, d_h)
                        t_coll.append(c_h)
                        t_clear.append(d_h_clamped)
                        v_mask.append(1.0)
                    else:
                        # Beyond episode: explicit mask 0.0 prevents silently treating missing labels as negative
                        t_coll.append(0.0)
                        t_clear.append(0.0)
                        v_mask.append(0.0)

                self.samples.append({
                    "warmup_obs": torch.tensor(w_obs, dtype=torch.float32),
                    "warmup_act": torch.tensor(w_act, dtype=torch.float32),
                    "future_act": torch.tensor(f_act, dtype=torch.float32),
                    "targets_coll": torch.tensor(t_coll, dtype=torch.float32),
                    "targets_clear": torch.tensor(t_clear, dtype=torch.float32),
                    "validity_mask": torch.tensor(v_mask, dtype=torch.float32),
                })

        print(f"Dataset constructed: {len(self.samples):,d} conditioning samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_training_and_calibration():
    print("=" * 80)
    print("SAWM - Milestone 5 Safety Heads Training & Calibration Pipeline")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Manifest
    manifest_path = "data/dataset_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_eps = manifest["partitions"]["train"]["episodes"]
    cal_eps = manifest["partitions"]["calibration"]["episodes"]
    test_eps = manifest["partitions"]["test"]["episodes"]

    print(f"Partitions: Train={len(train_eps)} ep | Cal={len(cal_eps)} ep | Test={len(test_eps)} ep (UNTOUCHED)")
    assert len(train_eps) == 106
    assert len(cal_eps) == 17
    assert len(test_eps) == 17

    # 2. Load Normalizer & Frozen M4 RSSM
    normalizer = TrajectoryNormalizer()
    normalizer.load("checkpoints/normalizer.pt")
    rssm = RSSM().to(device)
    rssm_ckpt = torch.load("checkpoints/rssm_best.pt", map_location=device, weights_only=False)
    rssm.load_state_dict(rssm_ckpt["model_state_dict"])
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad = False
    print("Frozen M4 RSSM loaded (199,089 parameters, requires_grad=False).")

    # 3. Build Datasets
    train_dataset = RolloutSafetyDataset(train_eps, normalizer, stride_nominal=4, stride_collision=1)
    cal_dataset = RolloutSafetyDataset(cal_eps, normalizer, stride_nominal=4, stride_collision=1)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False, drop_last=False)

    # 4. Initialize PhysicalRiskHead
    head = PhysicalRiskHead(latent_dim=160).to(device)
    breakdown = head.get_parameter_count_breakdown()
    print(f"PhysicalRiskHead initialized with {breakdown['total']:,d} trainable parameters.")
    assert breakdown["total"] == 31882

    # 5. Training Setup
    epochs = 20
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_cal_loss = float("inf")
    best_weights = None
    training_history = []

    print("\n--- Training PhysicalRiskHead on Train Partition (106 Episodes) ---")
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        head.train()
        total_train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            w_obs = batch["warmup_obs"].to(device)
            w_act = batch["warmup_act"].to(device)
            f_act = batch["future_act"].to(device)
            t_coll = batch["targets_coll"].to(device)
            t_clear = batch["targets_clear"].to(device)
            mask = batch["validity_mask"].to(device)

            # Frozen RSSM warmup & imagination
            with torch.no_grad():
                warm_out = rssm(w_obs, w_act)
                h0 = warm_out["deterministic_states"][:, -1]
                s0 = warm_out["stochastic_states"][:, -1]
                _, img_h, img_s = rssm.imagine(h0, s0, f_act)
                # Sliced at exact horizons [1, 5, 10, 20, 50]
                z_rollout = torch.cat([img_h, img_s], dim=-1)   # (B, 50, 160)
                z_horizons = z_rollout[:, HORIZON_INDICES, :]    # (B, 5, 160)

            # Forward pass through PhysicalRiskHead
            pred = head(z_horizons)
            loss, metrics = head.compute_loss(pred, t_coll, t_clear, validity_mask=mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, train_batches)

        # Evaluate on Calibration partition
        head.eval()
        total_cal_loss = 0.0
        cal_batches = 0

        with torch.no_grad():
            for batch in cal_loader:
                w_obs = batch["warmup_obs"].to(device)
                w_act = batch["warmup_act"].to(device)
                f_act = batch["future_act"].to(device)
                t_coll = batch["targets_coll"].to(device)
                t_clear = batch["targets_clear"].to(device)
                mask = batch["validity_mask"].to(device)

                warm_out = rssm(w_obs, w_act)
                h0 = warm_out["deterministic_states"][:, -1]
                s0 = warm_out["stochastic_states"][:, -1]
                _, img_h, img_s = rssm.imagine(h0, s0, f_act)
                z_rollout = torch.cat([img_h, img_s], dim=-1)
                z_horizons = z_rollout[:, HORIZON_INDICES, :]

                pred = head(z_horizons)
                c_loss, _ = head.compute_loss(pred, t_coll, t_clear, validity_mask=mask)
                total_cal_loss += c_loss.item()
                cal_batches += 1

        avg_cal_loss = total_cal_loss / max(1, cal_batches)
        current_lr = scheduler.get_last_lr()[0]

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "cal_loss": avg_cal_loss,
            "lr": current_lr,
        })

        if avg_cal_loss < best_cal_loss:
            best_cal_loss = avg_cal_loss
            best_weights = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            save_mark = " [*BEST]"
        else:
            save_mark = ""

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Cal Loss: {avg_cal_loss:.4f} | LR: {current_lr:.6f}{save_mark}")

    train_duration = time.perf_counter() - start_time
    print(f"\nTraining completed in {train_duration:.1f}s ({train_duration/60:.2f} min). Best Cal Loss: {best_cal_loss:.4f}")

    # Load best weights
    head.load_state_dict(best_weights)

    # -------------------------------------------------------------------------
    # 6. Temperature Scaling & Threshold Calibration on Calibration Partition
    # -------------------------------------------------------------------------
    print("\n--- Calibration on Held-Out Calibration Partition (17 Episodes) ---")
    head.eval()

    all_logits = []
    all_targets = []
    all_masks = []
    all_clear_preds = []
    all_clear_targets = []

    with torch.no_grad():
        for batch in cal_loader:
            w_obs = batch["warmup_obs"].to(device)
            w_act = batch["warmup_act"].to(device)
            f_act = batch["future_act"].to(device)
            t_coll = batch["targets_coll"].to(device)
            t_clear = batch["targets_clear"].to(device)
            mask = batch["validity_mask"].to(device)

            warm_out = rssm(w_obs, w_act)
            h0 = warm_out["deterministic_states"][:, -1]
            s0 = warm_out["stochastic_states"][:, -1]
            _, img_h, img_s = rssm.imagine(h0, s0, f_act)
            z_rollout = torch.cat([img_h, img_s], dim=-1)
            z_horizons = z_rollout[:, HORIZON_INDICES, :]

            pred = head(z_horizons)
            all_logits.append(pred["collision_logits"].cpu())
            all_targets.append(t_coll.cpu())
            all_masks.append(mask.cpu())
            all_clear_preds.append(pred["clearances"].cpu())
            all_clear_targets.append(t_clear.cpu())

    cal_logits = torch.cat(all_logits, dim=0).numpy()       # (N_cal, 5)
    cal_targets = torch.cat(all_targets, dim=0).numpy()     # (N_cal, 5)
    cal_masks = torch.cat(all_masks, dim=0).numpy()         # (N_cal, 5)
    cal_clear_preds = torch.cat(all_clear_preds, dim=0).numpy()
    cal_clear_targets = torch.cat(all_clear_targets, dim=0).numpy()

    # Temperature Scaling on Primary Horizon H=10 (index 2)
    h10_idx = 2
    valid_h10 = (cal_masks[:, h10_idx] > 0.5)
    logits_h10 = cal_logits[valid_h10, h10_idx]
    targets_h10 = cal_targets[valid_h10, h10_idx]

    def nll_obj(temp):
        t_val = max(1e-2, float(temp))
        scaled_logits = logits_h10 / t_val
        p = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30.0, 30.0)))
        eps = 1e-7
        p = np.clip(p, eps, 1.0 - eps)
        bce = - (targets_h10 * np.log(p) + (1.0 - targets_h10) * np.log(1.0 - p))
        return float(np.mean(bce))

    # 1D search for optimal temperature T in [0.1, 5.0]
    best_temp = 1.0
    best_nll = float("inf")
    for t_cand in np.linspace(0.1, 5.0, 500):
        val = nll_obj(t_cand)
        if val < best_nll:
            best_nll = val
            best_temp = float(t_cand)

    # Local refinement around best_temp
    for t_cand in np.linspace(max(0.05, best_temp - 0.05), best_temp + 0.05, 100):
        val = nll_obj(t_cand)
        if val < best_nll:
            best_nll = val
            best_temp = float(t_cand)

    optimal_temperature = float(best_temp)
    print(f"Optimal Temperature Scaling T: {optimal_temperature:.4f} (NLL: {best_nll:.4f})")

    # Apply optimal temperature
    scaled_probs_h10 = 1.0 / (1.0 + np.exp(-np.clip(logits_h10 / optimal_temperature, -30.0, 30.0)))

    # Threshold Optimization for Collision Warning Decision (H=10)
    best_f1 = 0.0
    best_threshold = 0.50
    best_rec = 0.0
    best_prec = 0.0

    for tau_cand in np.arange(0.10, 0.90, 0.02):
        preds_cand = (scaled_probs_h10 >= tau_cand).astype(np.float32)
        tp = np.sum((preds_cand == 1.0) & (targets_h10 == 1.0))
        fp = np.sum((preds_cand == 1.0) & (targets_h10 == 0.0))
        fn = np.sum((preds_cand == 0.0) & (targets_h10 == 1.0))

        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)

        if f1 > best_f1 and rec >= 0.85:
            best_f1 = f1
            best_threshold = float(tau_cand)
            best_rec = float(rec)
            best_prec = float(prec)

    # Brier score
    brier_score = float(np.mean((scaled_probs_h10 - targets_h10) ** 2))

    # Clearance MAE on calibration set
    clear_mae_all = float(np.mean(np.abs(cal_clear_preds[valid_h10, h10_idx] - cal_clear_targets[valid_h10, h10_idx])))
    danger_mask = (cal_clear_targets[valid_h10, h10_idx] < 0.08)
    if np.sum(danger_mask) > 0:
        clear_mae_danger = float(np.mean(np.abs(cal_clear_preds[valid_h10, h10_idx][danger_mask] - cal_clear_targets[valid_h10, h10_idx][danger_mask])))
    else:
        clear_mae_danger = 0.0

    print(f"Calibration Results (H=10, 0.10s lookahead):")
    print(f"  Calibrated Threshold tau_coll: {best_threshold:.2f}")
    print(f"  Calibration Recall:           {best_rec*100:.1f}%")
    print(f"  Calibration Precision:        {best_prec*100:.1f}%")
    print(f"  Calibration F1 Score:         {best_f1:.4f}")
    print(f"  Calibration Brier Score:      {brier_score:.4f}")
    print(f"  Calibration Clearance MAE:    {clear_mae_all*1000:.1f} mm (Danger zone: {clear_mae_danger*1000:.1f} mm)")

    # Update buffers in head
    head.temperature.fill_(optimal_temperature)
    head.collision_threshold.fill_(best_threshold)

    # -------------------------------------------------------------------------
    # 7. Save Checkpoint & Calibration Metadata
    # -------------------------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/physical_head_best.pt"
    torch.save({
        "model_state_dict": head.state_dict(),
        "temperature": optimal_temperature,
        "collision_threshold": best_threshold,
        "horizons": HORIZONS,
        "parameter_count": breakdown["total"],
        "parameter_breakdown": breakdown,
        "training_epochs": epochs,
        "best_cal_loss": best_cal_loss,
        "training_duration_seconds": train_duration,
    }, ckpt_path)
    print(f"\nPhysicalRiskHead checkpoint saved to: {ckpt_path} ({os.path.getsize(ckpt_path)/1024:.1f} KB)")

    cal_meta_path = "checkpoints/safety_calibrator.json"
    cal_metadata = {
        "temperature": optimal_temperature,
        "collision_threshold": best_threshold,
        "danger_distance": 0.08,
        "warning_distance": 0.20,
        "background_distance": 0.50,
        "calibration_metrics_h10": {
            "brier_score": brier_score,
            "f1": best_f1,
            "recall": best_rec,
            "precision": best_prec,
            "clearance_mae_mm": clear_mae_all * 1000,
            "clearance_mae_danger_mm": clear_mae_danger * 1000,
        },
        "training_history": training_history,
    }
    with open(cal_meta_path, "w") as f:
        json.dump(cal_metadata, f, indent=2)
    print(f"Safety calibrator metadata saved to: {cal_meta_path}")

    print("\n" + "=" * 80)
    print("TRAINING & CALIBRATION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_training_and_calibration()
