"""
Integrated Predictive Safety System for SAWM (Milestone 5 Requirement).

Unifies:
1. Learned Action-Conditioned PhysicalRiskHead (latent rollout evaluation).
2. Interpretable Multi-Body PredictiveContextualHarmHead (decoded kinematics & FK).

Provides:
- Non-interfering forward-looking safety evaluation across horizons H in {1, 5, 10, 20, 50}.
- Calibrated continuous safety score: S_total^(H) = max(R_phys^(H), C_harm^(H)) in [0, 1].
- 3-tier severity ratings: SAFE < WARNING < DANGER.
- Explainable alarm attribution.
- Strictly passive observer: Never alters commands or intervenes in planner execution.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn

from safety.predictive_physical_head import PhysicalRiskHead, HORIZONS
from safety.predictive_contextual_head import PredictiveContextualHarmHead


SEVERITY_ORDER = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
SEVERITY_NAMES = {0: "SAFE", 1: "WARNING", 2: "DANGER"}


class PredictiveSafetySystem:
    """
    Centralized, non-interfering predictive safety monitor.
    Evaluates forward action rollouts using the world model and safety heads.
    """

    def __init__(
        self,
        physical_head: PhysicalRiskHead,
        contextual_head: PredictiveContextualHarmHead,
        danger_distance: float = 0.08,
        warning_distance: float = 0.20,
        background_distance: float = 0.50,
        collision_threshold: float = 0.50,
    ):
        self.physical_head = physical_head
        self.contextual_head = contextual_head
        self.danger_distance = danger_distance
        self.warning_distance = warning_distance
        self.background_distance = background_distance
        self.collision_threshold = collision_threshold
        self.horizons = HORIZONS

    def compute_physical_risk_score(
        self,
        p_coll: float,
        d_min: float,
        threshold: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        Derive continuous physical risk score and discrete state from
        predicted collision probability and minimum geometric clearance.
        """
        tau = threshold if threshold is not None else self.collision_threshold

        if p_coll >= tau:
            return 1.0, "DANGER"
        elif d_min <= self.danger_distance:
            ratio = 1.0 - max(0.0, d_min) / max(1e-4, self.danger_distance)
            return float(0.8 + 0.2 * ratio), "DANGER"
        elif d_min < self.warning_distance:
            ratio = 1.0 - (d_min - self.danger_distance) / max(1e-4, (self.warning_distance - self.danger_distance))
            return float(0.2 + 0.6 * ratio), "WARNING"
        elif d_min < self.background_distance:
            ratio = (self.background_distance - d_min) / max(1e-4, (self.background_distance - self.warning_distance))
            return float(0.2 * ratio), "SAFE"
        else:
            return 0.0, "SAFE"

    def evaluate_plan(
        self,
        latent_rollout: Union[torch.Tensor, np.ndarray],
        decoded_obs_rollout: Union[torch.Tensor, np.ndarray],
        planned_fsm_stages: List[str],
        horizons: Optional[List[int]] = None,
        use_calibrated_probs: bool = True,
        normalizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full imagined rollout across both physical and contextual heads.

        Args:
            latent_rollout: Latent states (B, 5, 160) or (B, 50, 160) or (50, 160)
            decoded_obs_rollout: Predicted observation rollout (H, 49) or (1, H, 49)
            planned_fsm_stages: List[str] of planned stages for each future step
            horizons: Horizons to evaluate (default [1, 5, 10, 20, 50])
            use_calibrated_probs: Whether to use temperature-scaled collision probabilities
            normalizer: Optional normalizer to denormalize decoded observations

        Returns:
            Structured dictionary with per-horizon risk assessments and alarms.
        """
        if horizons is None:
            horizons = self.horizons

        # ---------------------------------------------------------------------
        # 1. Physical Head Inference
        # ---------------------------------------------------------------------
        if not isinstance(latent_rollout, torch.Tensor):
            latent_t = torch.tensor(latent_rollout, dtype=torch.float32)
        else:
            latent_t = latent_rollout

        if latent_t.dim() == 2:
            latent_t = latent_t.unsqueeze(0)  # (1, steps, 160)

        device = next(self.physical_head.parameters()).device
        latent_t = latent_t.to(device)

        self.physical_head.eval()
        with torch.no_grad():
            phys_out = self.physical_head(latent_t, use_temperature=use_calibrated_probs)

        probs_key = "calibrated_probs" if use_calibrated_probs else "collision_probs"
        probs_all = phys_out[probs_key][0].cpu().numpy()     # (5,)
        clearances_all = phys_out["clearances"][0].cpu().numpy() # (5,)

        # ---------------------------------------------------------------------
        # 2. Contextual Head Inference
        # ---------------------------------------------------------------------
        ctx_out = self.contextual_head.evaluate_rollout(
            predicted_obs=decoded_obs_rollout,
            planned_fsm_stages=planned_fsm_stages,
            horizons=horizons,
            normalizer=normalizer,
        )

        # ---------------------------------------------------------------------
        # 3. Multi-Horizon Unified Telemetry Fusion
        # ---------------------------------------------------------------------
        horizon_telemetry = {}
        max_overall_score = 0.0
        worst_state = "SAFE"
        all_alarms = []

        h_idx_map = {1: 0, 5: 1, 10: 2, 20: 3, 50: 4}

        for h in horizons:
            idx = h_idx_map[h]
            p_coll_h = float(probs_all[idx])
            d_min_h = float(clearances_all[idx])

            r_phys, state_phys = self.compute_physical_risk_score(p_coll_h, d_min_h)

            ctx_h_data = ctx_out.get(h, {"harm_score": 0.0, "state": "SAFE", "alarms": []})
            c_harm = float(ctx_h_data["harm_score"])
            state_ctx = ctx_h_data["state"]

            # Unified continuous score
            s_total = max(r_phys, c_harm)

            # Strict total ordering for severity
            sev_phys = SEVERITY_ORDER[state_phys]
            sev_ctx = SEVERITY_ORDER[state_ctx]
            sev_overall = max(sev_phys, sev_ctx)
            state_overall = SEVERITY_NAMES[sev_overall]

            # Alarms
            h_alarms = []
            if p_coll_h >= self.collision_threshold:
                h_alarms.append(
                    f"PREDICTED COLLISION at H={h} (dt={h*0.01:.2f}s, P={p_coll_h:.2f} >= tau={self.collision_threshold:.2f}, clearance={d_min_h*1000:.1f}mm)"
                )
            elif d_min_h <= self.danger_distance:
                h_alarms.append(
                    f"CRITICAL PROXIMITY at H={h} (dt={h*0.01:.2f}s, clearance={d_min_h*1000:.1f}mm <= {self.danger_distance*1000:.0f}mm)"
                )
            elif d_min_h < self.warning_distance:
                h_alarms.append(
                    f"Proximity Warning at H={h} (clearance={d_min_h*1000:.1f}mm)"
                )

            h_alarms.extend(ctx_h_data.get("alarms", []))

            horizon_telemetry[h] = {
                "horizon": h,
                "dt_seconds": h * 0.01,
                "collision_prob": p_coll_h,
                "geometric_clearance": d_min_h,
                "physical_risk_score": r_phys,
                "physical_state": state_phys,
                "contextual_harm_score": c_harm,
                "contextual_state": state_ctx,
                "overall_score": s_total,
                "overall_state": state_overall,
                "alarms": h_alarms,
            }

            if s_total > max_overall_score:
                max_overall_score = s_total
            if sev_overall > SEVERITY_ORDER[worst_state]:
                worst_state = state_overall
            all_alarms.extend(h_alarms)

        return {
            "horizons": horizon_telemetry,
            "max_overall_score": float(max_overall_score),
            "worst_state": worst_state,
            "all_alarms": list(dict.fromkeys(all_alarms)),
            "default_evaluation": horizon_telemetry.get(10, {}),  # H=10 primary early warning horizon
        }
