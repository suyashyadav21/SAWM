"""
Trainer Engine for SAWM RSSM World Model.

Features:
- AdamW optimizer with cosine learning rate scheduling
- Gradient norm clipping
- KL balancing with separate representation/dynamics updates
- Feature-weighted reconstruction + geodesic orientation + binary contact losses
- Validation evaluation and best checkpoint saving
- Progress telemetry and loss history tracking
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from world_model.rssm import RSSM


class RSSMTrainer:
    """Manages training and validation loops for the RSSM."""

    def __init__(
        self,
        model: RSSM,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        train_cfg = config.get("training", {})
        self.lr = float(train_cfg.get("learning_rate", 0.0003))
        self.weight_decay = float(train_cfg.get("weight_decay", 0.0001))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 100.0))
        self.epochs = int(train_cfg.get("epochs", 40))
        self.kl_scale = float(train_cfg.get("kl_scale", 0.10))
        self.kl_balance = float(train_cfg.get("kl_balance", 0.80))
        self.free_nats = float(train_cfg.get("free_nats", 1.0))
        self.loss_weights = config.get("loss_weights", {})

        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        self.checkpoint_name = train_cfg.get("checkpoint_name", "rssm_best.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
            "val_rot_err_rad": [],
        }

    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch across all batches."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        batch_count = 0

        for batch in self.train_loader:
            obs = batch["observations"].to(self.device)      # (B, T, 49)
            act = batch["actions"].to(self.device)           # (B, T-1, 8)

            self.optimizer.zero_grad()
            preds = self.model(obs, act)
            losses = self.model.compute_loss(
                targets=obs,
                predictions=preds,
                kl_scale=self.kl_scale,
                kl_balance=self.kl_balance,
                free_nats=self.free_nats,
                loss_weights=self.loss_weights,
            )

            losses["loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += float(losses["loss"].item())
            total_recon += float(losses["recon_loss"].item())
            total_kl += float(losses["kl_loss"].item())
            batch_count += 1

            if batch_count % 100 == 0 or batch_count == len(self.train_loader):
                print(f"  [Train] Batch {batch_count:3d}/{len(self.train_loader):3d} | Loss: {losses['loss'].item():6.3f} | Recon: {losses['recon_loss'].item():6.3f} | KL: {losses['kl_loss'].item():5.2f}", flush=True)

        return {
            "train_loss": total_loss / max(1, batch_count),
            "train_recon": total_recon / max(1, batch_count),
            "train_kl": total_kl / max(1, batch_count),
        }

    def validate(self) -> Dict[str, float]:
        """Evaluate model on held-out validation sequences."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_rot = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                obs = batch["observations"].to(self.device)
                act = batch["actions"].to(self.device)

                preds = self.model(obs, act)
                losses = self.model.compute_loss(
                    targets=obs,
                    predictions=preds,
                    kl_scale=self.kl_scale,
                    kl_balance=self.kl_balance,
                    free_nats=self.free_nats,
                    loss_weights=self.loss_weights,
                )

                total_loss += float(losses["loss"].item())
                total_recon += float(losses["recon_loss"].item())
                total_kl += float(losses["kl_loss"].item())
                total_rot += float(losses["ee_rot_loss_rad"].item() + losses["obj_rot_loss_rad"].item())
                batch_count += 1

        return {
            "val_loss": total_loss / max(1, batch_count),
            "val_recon": total_recon / max(1, batch_count),
            "val_kl": total_kl / max(1, batch_count),
            "val_rot_err_rad": total_rot / max(1, batch_count),
        }

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """Run full training loop with checkpointing."""
        best_val_loss = float("inf")
        best_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        if verbose:
            print("=" * 75)
            print(f"Beginning RSSM Training ({self.epochs} epochs) on {self.device}")
            print(f"Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}")
            print("=" * 75)

        t_start = time.time()
        for epoch in range(1, self.epochs + 1):
            t_ep = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            ep_time = time.time() - t_ep

            # Record history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_recon"].append(train_metrics["train_recon"])
            self.history["train_kl"].append(train_metrics["train_kl"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_recon"].append(val_metrics["val_recon"])
            self.history["val_kl"].append(val_metrics["val_kl"])
            self.history["val_rot_err_rad"].append(val_metrics["val_rot_err_rad"])

            # Checkpoint best model
            is_best = val_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["val_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": self.config,
                }, best_path)

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == self.epochs or is_best):
                star = " *" if is_best else "  "
                print(
                    f"Epoch {epoch:2d}/{self.epochs:2d} [{ep_time:4.1f}s]{star} | "
                    f"TrLoss: {train_metrics['train_loss']:7.3f} (Recon: {train_metrics['train_recon']:6.3f}, KL: {train_metrics['train_kl']:5.2f}) | "
                    f"ValLoss: {val_metrics['val_loss']:7.3f} (Recon: {val_metrics['val_recon']:6.3f}, RotErr: {val_metrics['val_rot_err_rad']*180/np.pi:4.1f} deg)",
                    flush=True
                )

        total_time = time.time() - t_start
        if verbose:
            print("=" * 75, flush=True)
            print(f"Training completed in {total_time:.1f}s | Best Val Loss: {best_val_loss:.4f}", flush=True)
            print(f"Best checkpoint saved to: {best_path}", flush=True)
            print("=" * 75, flush=True)

        return self.history
