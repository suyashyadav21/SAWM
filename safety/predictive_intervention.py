"""
Predictive Intervention Pipeline for SAWM (Milestone 6 Step 2 MVP).

Connects:
1. M4 RSSM (causal online state tracking + action-conditioned forward rollout)
2. M4 Normalizer (feature-wise observation normalization)
3. M5 PhysicalRiskHead (calibrated collision probability P10 and clearance d10)
4. M6 Step 1 SafetySupervisor (arbitration, speed scaling, and dynamic arrest)
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch

from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import TrajectoryNormalizer, ObservationBuilder
from safety.predictive_physical_head import PhysicalRiskHead
from safety.safety_supervisor import SafetySupervisor, TacticalTelemetry, ReactiveTelemetry, SupervisorState, InterventionCommand


class H10Predictor:
    """
    Online H=10 predictive risk evaluator.
    Maintains RSSM latent state (h_t, s_t) online and projects forward
    10 steps under nominal planned actions to produce validated TacticalTelemetry.
    """

    def __init__(
        self,
        rssm_path: str = "checkpoints/rssm_best.pt",
        normalizer_path: str = "checkpoints/normalizer.pt",
        physical_head_path: str = "checkpoints/physical_head_best.pt",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.rssm_path = rssm_path
        self.normalizer_path = normalizer_path
        self.physical_head_path = physical_head_path

        # 1. Load RSSM
        if not os.path.exists(rssm_path):
            raise FileNotFoundError(f"RSSM checkpoint not found: {rssm_path}")
        rssm_ckpt = torch.load(rssm_path, map_location=self.device, weights_only=False)
        cfg = rssm_ckpt.get("config", {})
        m_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
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
        self.rssm = RSSM(rssm_cfg).to(self.device)
        self.rssm.load_state_dict(rssm_ckpt["model_state_dict"])
        self.rssm.eval()

        # 2. Load TrajectoryNormalizer
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer checkpoint not found: {normalizer_path}")
        norm_ckpt = torch.load(normalizer_path, map_location=self.device, weights_only=False)
        self.normalizer = TrajectoryNormalizer(obs_dim=49)
        self.normalizer.mean = norm_ckpt["mean"].cpu().numpy() if isinstance(norm_ckpt["mean"], torch.Tensor) else np.array(norm_ckpt["mean"])
        self.normalizer.std = norm_ckpt["std"].cpu().numpy() if isinstance(norm_ckpt["std"], torch.Tensor) else np.array(norm_ckpt["std"])
        self.normalizer.exempt_indices = set(norm_ckpt["exempt_indices"])
        self.normalizer.is_fit = bool(norm_ckpt["is_fit"])

        # 3. Load PhysicalRiskHead
        if not os.path.exists(physical_head_path):
            raise FileNotFoundError(f"Physical risk head checkpoint not found: {physical_head_path}")
        head_ckpt = torch.load(physical_head_path, map_location=self.device, weights_only=False)
        self.physical_head = PhysicalRiskHead(latent_dim=160).to(self.device)
        self.physical_head.load_state_dict(head_ckpt["model_state_dict"])
        if "temperature" in head_ckpt:
            self.physical_head.temperature.copy_(torch.as_tensor(head_ckpt["temperature"], device=self.device))
        if "collision_threshold" in head_ckpt:
            self.physical_head.collision_threshold.copy_(torch.as_tensor(head_ckpt["collision_threshold"], device=self.device))
        self.physical_head.eval()

        # Online runtime state
        self.obs_builder: Optional[ObservationBuilder] = None
        self.h_t: Optional[torch.Tensor] = None
        self.s_t: Optional[torch.Tensor] = None
        self.current_time: float = 0.0

    def reset(self, env: Any, raw_obs: Dict[str, Any], current_time: float = 0.0) -> None:
        """Initialize observation builder and extract initial posterior mean."""
        self.obs_builder = ObservationBuilder(env)
        obs_vec = self.obs_builder.extract_observation_vector(raw_obs)
        norm_obs = self.normalizer.normalize(obs_vec)
        obs_tensor = torch.from_numpy(norm_obs).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.h_t, self.s_t, _ = self.rssm.initial_posterior_mean(obs_tensor)

        self.current_time = float(current_time)

    def update(self, executed_action_8d: np.ndarray, raw_obs: Dict[str, Any], current_time: float) -> None:
        """Causal update of RSSM latent state with executed action and current observation."""
        if self.obs_builder is None:
            raise RuntimeError("H10Predictor must be reset() before calling update()")

        obs_vec = self.obs_builder.extract_observation_vector(raw_obs)
        norm_obs = self.normalizer.normalize(obs_vec)
        obs_tensor = torch.from_numpy(norm_obs).float().unsqueeze(0).to(self.device)
        act_tensor = torch.from_numpy(executed_action_8d).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.h_t, self.s_t, _ = self.rssm.step_causal_mean(
                self.s_t, act_tensor, self.h_t, obs_tensor
            )

        self.current_time = float(current_time)

    def predict_multi_horizon(
        self,
        planned_actions: np.ndarray,
        current_time: Optional[float] = None,
        raw_obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Roll out latent state 50 steps forward under planned actions and predict risk across all horizons:
        H in {1, 5, 10, 20, 50} extracted from the exact same 50-step imagined action rollout.
        """
        t_curr = self.current_time if current_time is None else float(current_time)

        # Telemetry validation guard
        if not np.isfinite(t_curr) or planned_actions is None:
            return {"valid": False}

        if not isinstance(planned_actions, np.ndarray) or planned_actions.ndim != 2 or planned_actions.shape[-1] != 8:
            return {"valid": False}

        if not np.all(np.isfinite(planned_actions)):
            return {"valid": False}

        if self.h_t is None or self.s_t is None:
            return {"valid": False}

        try:
            # Ensure planned actions has exactly 50 steps
            if planned_actions.shape[0] < 50:
                pad_len = 50 - planned_actions.shape[0]
                last_act = planned_actions[-1:]
                act_50 = np.concatenate([planned_actions, np.repeat(last_act, pad_len, axis=0)], axis=0)
            else:
                act_50 = planned_actions[:50]

            with torch.no_grad():
                act_tensor = torch.from_numpy(act_50).float().unsqueeze(0).to(self.device)
                _, imag_h, imag_s = self.rssm.imagine(self.h_t, self.s_t, act_tensor)
                z = torch.cat([imag_h, imag_s], dim=-1)  # (1, 50, 160)

                out = self.physical_head(z, use_temperature=True)
                probs = out["calibrated_probs"][0].detach().cpu().numpy()
                clearances = out["clearances"][0].detach().cpu().numpy()

            # Contextual metrics extraction
            human_proxy_dist = 999.0
            c_harm = 0.0
            if raw_obs is not None:
                if "hazards" in raw_obs and "robot" in raw_obs:
                    proxy_pos = np.array(raw_obs["hazards"]["proxy_pos"])
                    ee_pos = np.array(raw_obs["robot"]["ee_position"])
                    human_proxy_dist = float(np.linalg.norm(ee_pos - proxy_pos))

            res = {
                "valid": True,
                "timestamp": t_curr,
                "horizons": [1, 5, 10, 20, 50],
                "probs": probs,
                "clearances": clearances,
                "p1": float(probs[0]),
                "p5": float(probs[1]),
                "p10": float(probs[2]),
                "p20": float(probs[3]),
                "p50": float(probs[4]),
                "d1_clear": float(clearances[0]),
                "d5_clear": float(clearances[1]),
                "d10_clear": float(clearances[2]),
                "d20_clear": float(clearances[3]),
                "d50_clear": float(clearances[4]),
                "human_proxy_distance": human_proxy_dist,
                "c_harm": c_harm,
            }
            self.last_multi_horizon = res
            return res
        except Exception:
            return {"valid": False}

    def predict_h10(
        self,
        planned_actions: np.ndarray,
        current_time: Optional[float] = None,
        raw_obs: Optional[Dict[str, Any]] = None,
    ) -> TacticalTelemetry:
        """
        Roll out latent state 50 steps forward under planned actions and extract H=10 tactical risk.
        Ensures H=1, 5, 10, 20, 50 are all extracted from the SAME 50-step imagined action rollout.
        
        Args:
            planned_actions: (50, 8) or (10, 8) planned future actions in RSSM standard format.
            current_time: Optional explicit timestamp; defaults to self.current_time.
            raw_obs: Optional raw observation dictionary for contextual metrics.
            
        Returns:
            Validated TacticalTelemetry dataclass instance.
        """
        multi = self.predict_multi_horizon(planned_actions, current_time=current_time, raw_obs=raw_obs)
        if not multi.get("valid", False):
            return TacticalTelemetry(timestamp=float("nan"), p10=float("nan"))

        return TacticalTelemetry(
            p10=multi["p10"],
            d10_clear=multi["d10_clear"],
            c_harm=multi["c_harm"],
            closure_rate=0.0,
            human_proxy_distance=multi["human_proxy_distance"],
            timestamp=multi["timestamp"],
            age_ms=0.0,
        )
