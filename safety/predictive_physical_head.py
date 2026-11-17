"""
Action-Conditioned Physical Risk Head for SAWM (Milestone 5 Requirement).

Evaluates RSSM latent states rolled forward under an action sequence to predict:
1. Cumulative collision probabilities P_coll^(H) in [0, 1] for horizons H in {1, 5, 10, 20, 50}.
2. Cumulative minimum geometric clearances d_min^(H) in R^+ (meters) for horizons H in {1, 5, 10, 20, 50}.

Architecture:
- Input: 160-dimensional RSSM latent representation z = [h, s]
- Trunk: Linear(160->128) + LayerNorm(128) + GELU() + Dropout(0.05) -> Linear(128->64) + LayerNorm(64) + GELU()
- Collision Branch: Linear(64->5) with cumulative Softplus parameterization enforcing P^(1) <= P^(5) <= P^(10) <= P^(20) <= P^(50)
- Clearance Branch: Linear(64->32) + LayerNorm(32) + GELU() -> Linear(32->5) with reverse-cumulative Softplus parameterization
  enforcing d^(1) >= d^(5) >= d^(10) >= d^(20) >= d^(50)
- Verified parameter count: Exactly 31,882 parameters.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


HORIZONS = [1, 5, 10, 20, 50]
HORIZON_INDICES = [0, 4, 9, 19, 49]  # 0-indexed rollout steps corresponding to H
HORIZON_WEIGHTS = [0.20, 0.20, 0.25, 0.20, 0.15]


class PhysicalRiskHead(nn.Module):
    """
    Lightweight PyTorch MLP predicting multi-horizon cumulative collision probability
    and minimum clearance from RSSM latent states.
    """

    def __init__(self, latent_dim: int = 160, dropout_p: float = 0.05):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizons = HORIZONS
        self.num_horizons = len(HORIZONS)

        # ---------------------------------------------------------------------
        # Shared Trunk (29,248 params)
        # ---------------------------------------------------------------------
        self.trunk1 = nn.Linear(latent_dim, 128)        # 160 * 128 + 128 = 20,608
        self.ln1 = nn.LayerNorm(128)                    # 128 + 128 = 256 -> Subtotal: 20,864
        self.dropout = nn.Dropout(dropout_p)
        self.trunk2 = nn.Linear(128, 64)                # 128 * 64 + 64 = 8,256
        self.ln2 = nn.LayerNorm(64)                     # 64 + 64 = 128 -> Subtotal: 8,384

        # ---------------------------------------------------------------------
        # Collision Probability Branch (325 params)
        # ---------------------------------------------------------------------
        self.coll_branch = nn.Linear(64, 5)             # 64 * 5 + 5 = 325

        # ---------------------------------------------------------------------
        # Clearance Regression Branch (2,309 params)
        # ---------------------------------------------------------------------
        self.clear1 = nn.Linear(64, 32)                 # 64 * 32 + 32 = 2,080
        self.ln_clear = nn.LayerNorm(32)                # 32 + 32 = 64 -> Subtotal: 2,144
        self.clear2 = nn.Linear(32, 5)                  # 32 * 5 + 5 = 165

        # Calibration parameters (not updated during backprop training)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("collision_threshold", torch.tensor(0.50, dtype=torch.float32))

    def get_parameter_count_breakdown(self) -> Dict[str, int]:
        """Return exact parameter counts per submodule."""
        return {
            "trunk_layer1": sum(p.numel() for p in self.trunk1.parameters()) + sum(p.numel() for p in self.ln1.parameters()),
            "trunk_layer2": sum(p.numel() for p in self.trunk2.parameters()) + sum(p.numel() for p in self.ln2.parameters()),
            "collision_branch": sum(p.numel() for p in self.coll_branch.parameters()),
            "clearance_layer1": sum(p.numel() for p in self.clear1.parameters()) + sum(p.numel() for p in self.ln_clear.parameters()),
            "clearance_layer2": sum(p.numel() for p in self.clear2.parameters()),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(
        self,
        z: torch.Tensor,
        use_temperature: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PhysicalRiskHead.

        Args:
            z: Latent tensor. Accepted shapes:
               - (B, 5, 160): Latent states at horizons [1, 5, 10, 20, 50].
               - (B, 50, 160): Full 50-step rollout; automatically sliced at HORIZON_INDICES.
               - (B, 160): Single latent; expanded across all 5 horizons.
            use_temperature: If True, scale logits by learned calibration temperature.

        Returns:
            Dict containing:
                - "collision_probs": (B, 5) monotonic cumulative collision probabilities in [0, 1]
                - "clearances": (B, 5) monotonic cumulative minimum clearances in meters (R^+)
                - "collision_logits": (B, 5) raw unscaled collision logits
                - "calibrated_probs": (B, 5) temperature-scaled collision probabilities
        """
        # Align input shape to (B, 5, 160)
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, 5, -1)
        elif z.dim() == 3:
            if z.size(1) == 5:
                pass
            elif z.size(1) == 1:
                z = z.expand(-1, 5, -1)
            else:
                # Map horizons [1, 5, 10, 20, 50] to available rollout steps
                indices = [min(h - 1, z.size(1) - 1) for h in self.horizons]
                z = z[:, indices, :]
        else:
            raise ValueError(f"Unexpected latent shape for PhysicalRiskHead: {z.shape}")

        batch_size = z.size(0)

        # Shared Trunk
        h = self.dropout(F.gelu(self.ln1(self.trunk1(z))))       # (B, 5, 128)
        h = F.gelu(self.ln2(self.trunk2(h)))                     # (B, 5, 64)

        # ---------------------------------------------------------------------
        # Collision Branch with Guaranteed Monotonicity
        # ---------------------------------------------------------------------
        raw_coll = self.coll_branch(h)                           # (B, 5, 5)
        # Extract diagonal: output i corresponds to horizon i
        coll_diag = torch.diagonal(raw_coll, dim1=1, dim2=2)     # (B, 5)

        l0 = coll_diag[:, 0]
        l1 = l0 + F.softplus(coll_diag[:, 1])
        l2 = l1 + F.softplus(coll_diag[:, 2])
        l3 = l2 + F.softplus(coll_diag[:, 3])
        l4 = l3 + F.softplus(coll_diag[:, 4])
        logits = torch.stack([l0, l1, l2, l3, l4], dim=-1)       # (B, 5)
        probs = torch.sigmoid(logits)                            # (B, 5)

        # Calibrated probabilities via temperature scaling
        cal_temp = self.temperature.clamp(min=1e-3)
        calibrated_probs = torch.sigmoid(logits / cal_temp)

        # ---------------------------------------------------------------------
        # Clearance Branch with Guaranteed Monotonicity (Reverse Cumulative)
        # ---------------------------------------------------------------------
        c = F.gelu(self.ln_clear(self.clear1(h)))                # (B, 5, 32)
        raw_clear = self.clear2(c)                               # (B, 5, 5)
        clear_diag = torch.diagonal(raw_clear, dim1=1, dim2=2)   # (B, 5)

        # Reverse cumulative: H=50 is base, shorter horizons accumulate positive deltas
        d4 = F.softplus(clear_diag[:, 4])
        d3 = d4 + F.softplus(clear_diag[:, 3])
        d2 = d3 + F.softplus(clear_diag[:, 2])
        d1 = d2 + F.softplus(clear_diag[:, 1])
        d0 = d1 + F.softplus(clear_diag[:, 0])
        clearances = torch.stack([d0, d1, d2, d3, d4], dim=-1)   # (B, 5)

        return {
            "collision_probs": probs,
            "clearances": clearances,
            "collision_logits": logits,
            "calibrated_probs": calibrated_probs if use_temperature else probs,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_coll: torch.Tensor,
        target_clear: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        alpha_focal: float = 0.75,
        gamma_focal: float = 2.0,
        lambda_coll: float = 2.0,
        lambda_clear: float = 1.5,
        huber_delta: float = 0.02,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-horizon training loss:
        L_total = sum_H w_H * [ lambda_coll * L_focal + lambda_clear * L_Huber(prox_weighted) ]

        Args:
            outputs: Output dictionary from self.forward(z)
            target_coll: (B, 5) binary cumulative collision targets {0, 1}
            target_clear: (B, 5) cumulative minimum clearances (meters, >= 0)
            validity_mask: (B, 5) binary validity mask (1 for valid horizons, 0 for missing)
            alpha_focal: Focal loss alpha balancing factor
            gamma_focal: Focal loss focusing parameter
            lambda_coll: Weight for collision loss
            lambda_clear: Weight for clearance loss
            huber_delta: Transition point for Huber loss (meters)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        probs = outputs["collision_probs"]          # (B, 5)
        clearances = outputs["clearances"]          # (B, 5)
        batch_size = probs.size(0)

        if validity_mask is None:
            validity_mask = torch.ones_like(probs)

        # ---------------------------------------------------------------------
        # 1. Focal Collision Loss
        # ---------------------------------------------------------------------
        eps = 1e-7
        p_clamped = probs.clamp(eps, 1.0 - eps)
        pt = target_coll * p_clamped + (1.0 - target_coll) * (1.0 - p_clamped)
        alpha_t = target_coll * alpha_focal + (1.0 - target_coll) * (1.0 - alpha_focal)
        focal_weight = alpha_t * ((1.0 - pt) ** gamma_focal)
        bce = - (target_coll * torch.log(p_clamped) + (1.0 - target_coll) * torch.log(1.0 - p_clamped))
        focal_loss = focal_weight * bce             # (B, 5)

        # ---------------------------------------------------------------------
        # 2. Proximity-Weighted Huber Clearance Loss
        # ---------------------------------------------------------------------
        # w_prox(d) = 1.0 + 3.0 * exp(-10.0 * d_target)
        w_prox = 1.0 + 3.0 * torch.exp(-10.0 * target_clear.clamp(min=0.0))
        # Huber / Smooth L1 loss
        clear_diff = torch.abs(clearances - target_clear)
        huber = torch.where(
            clear_diff < huber_delta,
            0.5 * (clear_diff ** 2) / huber_delta,
            clear_diff - 0.5 * huber_delta,
        )
        weighted_huber = w_prox * huber             # (B, 5)

        # ---------------------------------------------------------------------
        # 3. Horizon Aggregation with Explicit Validity Masking
        # ---------------------------------------------------------------------
        h_weights = torch.tensor(HORIZON_WEIGHTS, device=probs.device, dtype=probs.dtype)  # (5,)

        total_loss = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        per_h_coll_loss = []
        per_h_clear_loss = []

        for i, h in enumerate(self.horizons):
            mask_i = validity_mask[:, i]
            n_valid = mask_i.sum().clamp(min=1.0)

            l_coll_i = (focal_loss[:, i] * mask_i).sum() / n_valid
            l_clear_i = (weighted_huber[:, i] * mask_i).sum() / n_valid

            h_loss = h_weights[i] * (lambda_coll * l_coll_i + lambda_clear * l_clear_i)
            total_loss = total_loss + h_loss

            per_h_coll_loss.append(float(l_coll_i.detach()))
            per_h_clear_loss.append(float(l_clear_i.detach()))

        metrics = {
            "loss_total": float(total_loss.detach()),
            "loss_coll_h1": per_h_coll_loss[0],
            "loss_coll_h5": per_h_coll_loss[1],
            "loss_coll_h10": per_h_coll_loss[2],
            "loss_coll_h20": per_h_coll_loss[3],
            "loss_coll_h50": per_h_coll_loss[4],
            "loss_clear_h1": per_h_clear_loss[0],
            "loss_clear_h5": per_h_clear_loss[1],
            "loss_clear_h10": per_h_clear_loss[2],
            "loss_clear_h20": per_h_clear_loss[3],
            "loss_clear_h50": per_h_clear_loss[4],
        }

        return total_loss, metrics
