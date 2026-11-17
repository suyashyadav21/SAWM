"""
Recurrent State-Space Model (RSSM) for SAWM Franka Panda Manipulation.

Implements:
- Observation Encoder MLP
- Deterministic Recurrent Transition Model (GRUCell)
- Posterior Latent Estimator (Gaussian Representation Model)
- Prior Latent Predictor (Gaussian Transition Model)
- Observation Decoder with Unit Quaternion Normalization
- Open-Loop Multi-Step Latent Rollout Pipeline
- KL Balancing & Geodesic Orientation-Aware Loss
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


@dataclass
class RSSMConfig:
    """Hyperparameters and dimensions for RSSM."""
    obs_dim: int = 49
    action_dim: int = 8
    encoder_hidden_dim: int = 128
    deterministic_dim: int = 128
    stochastic_dim: int = 32
    prior_hidden_dim: int = 128
    posterior_hidden_dim: int = 128
    decoder_hidden_dim: int = 128
    min_std: float = 0.1
    max_std: float = 10.0


class ObservationEncoder(nn.Module):
    """Encodes 49-dim structured vector observation into latent representation e_t."""

    def __init__(self, obs_dim: int = 49, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RecurrentModel(nn.Module):
    """Deterministic state transition model using GRUCell."""

    def __init__(self, action_dim: int = 8, stochastic_dim: int = 32, deterministic_dim: int = 128):
        super().__init__()
        self.cell = nn.GRUCell(input_size=stochastic_dim + action_dim, hidden_size=deterministic_dim)

    def forward(self, prev_stochastic: torch.Tensor, prev_action: torch.Tensor, prev_deterministic: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prev_stochastic, prev_action], dim=-1)
        return self.cell(x, prev_deterministic)


class PosteriorEstimator(nn.Module):
    """Infers stochastic state distribution q(s_t | h_t, e_t) conditioned on history and observation."""

    def __init__(self, deterministic_dim: int = 128, encoder_dim: int = 128, hidden_dim: int = 128, stochastic_dim: int = 32, min_std: float = 0.1, max_std: float = 10.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.fc_hidden = nn.Sequential(
            nn.Linear(deterministic_dim + encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.fc_std = nn.Linear(hidden_dim, stochastic_dim)

    def forward(self, deterministic: torch.Tensor, encoded_obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        x = torch.cat([deterministic, encoded_obs], dim=-1)
        h = self.fc_hidden(x)
        mean = self.fc_mean(h)
        std = F.softplus(self.fc_std(h)) + self.min_std
        std = torch.clamp(std, max=self.max_std)
        dist = Normal(mean, std)
        sample = dist.rsample()
        return dist, sample


class PriorPredictor(nn.Module):
    """Predicts stochastic state transition p(s_t | h_t) conditioned solely on deterministic recurrent state."""

    def __init__(self, deterministic_dim: int = 128, hidden_dim: int = 128, stochastic_dim: int = 32, min_std: float = 0.1, max_std: float = 10.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

        self.fc_hidden = nn.Sequential(
            nn.Linear(deterministic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, stochastic_dim)
        self.fc_std = nn.Linear(hidden_dim, stochastic_dim)

    def forward(self, deterministic: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        h = self.fc_hidden(deterministic)
        mean = self.fc_mean(h)
        std = F.softplus(self.fc_std(h)) + self.min_std
        std = torch.clamp(std, max=self.max_std)
        dist = Normal(mean, std)
        sample = dist.rsample()
        return dist, sample


class ObservationDecoder(nn.Module):
    """Reconstructs 49-dim observation vector from combined latent z_t = [h_t, s_t]."""

    def __init__(self, deterministic_dim: int = 128, stochastic_dim: int = 32, hidden_dim: int = 128, obs_dim: int = 49):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, deterministic: torch.Tensor, stochastic: torch.Tensor) -> torch.Tensor:
        z = torch.cat([deterministic, stochastic], dim=-1)
        raw_out = self.net(z)

        # Normalize predicted quaternions onto unit 3-sphere:
        # EE quaternion: indices [18:22]
        # Object quaternion: indices [25:29]
        ee_quat_raw = raw_out[..., 18:22]
        ee_quat_norm = ee_quat_raw / torch.norm(ee_quat_raw, dim=-1, keepdim=True).clamp(min=1e-8)

        obj_quat_raw = raw_out[..., 25:29]
        obj_quat_norm = obj_quat_raw / torch.norm(obj_quat_raw, dim=-1, keepdim=True).clamp(min=1e-8)

        # Reconstruct with normalized quaternions
        out = torch.cat([
            raw_out[..., :18],
            ee_quat_norm,
            raw_out[..., 22:25],
            obj_quat_norm,
            raw_out[..., 29:],
        ], dim=-1)
        return out


class RSSM(nn.Module):
    """Complete Recurrent State-Space Model integrating all latent submodules."""

    def __init__(self, config: Optional[RSSMConfig] = None):
        super().__init__()
        self.cfg = config or RSSMConfig()

        self.encoder = ObservationEncoder(self.cfg.obs_dim, self.cfg.encoder_hidden_dim)
        self.recurrent = RecurrentModel(self.cfg.action_dim, self.cfg.stochastic_dim, self.cfg.deterministic_dim)
        self.posterior = PosteriorEstimator(
            self.cfg.deterministic_dim,
            self.cfg.encoder_hidden_dim,
            self.cfg.posterior_hidden_dim,
            self.cfg.stochastic_dim,
            self.cfg.min_std,
            self.cfg.max_std,
        )
        self.prior = PriorPredictor(
            self.cfg.deterministic_dim,
            self.cfg.prior_hidden_dim,
            self.cfg.stochastic_dim,
            self.cfg.min_std,
            self.cfg.max_std,
        )
        self.decoder = ObservationDecoder(
            self.cfg.deterministic_dim,
            self.cfg.stochastic_dim,
            self.cfg.decoder_hidden_dim,
            self.cfg.obs_dim,
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero initial state for (h_0, s_0)."""
        h_0 = torch.zeros(batch_size, self.cfg.deterministic_dim, device=device)
        s_0 = torch.zeros(batch_size, self.cfg.stochastic_dim, device=device)
        return h_0, s_0

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_h: Optional[torch.Tensor] = None,
        initial_s: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward filtering over a sequence of length T.
        
        Args:
            observations: (B, T, 49)
            actions: (B, T-1, 8)
            
        Returns:
            Dictionary of posterior distributions, prior distributions,
            deterministic states h, stochastic states s, and reconstructed observations.
        """
        B, T, _ = observations.shape
        device = observations.device

        h_t, s_t = (
            (initial_h, initial_s)
            if initial_h is not None and initial_s is not None
            else self.initial_state(B, device)
        )

        h_list = [h_t]
        s_post_list = [s_t]
        post_dists = []
        prior_dists = []

        # Encode initial observation
        e_0 = self.encoder(observations[:, 0])
        post_dist_0, s_0 = self.posterior(h_t, e_0)
        s_post_list[0] = s_0
        post_dists.append(post_dist_0)

        # Initial prior matches posterior for t=0
        prior_dists.append(post_dist_0)

        for t in range(1, T):
            a_prev = actions[:, t - 1]
            s_prev = s_post_list[-1]

            # 1. Deterministic step
            h_t = self.recurrent(s_prev, a_prev, h_t)
            h_list.append(h_t)

            # 2. Prior prediction (imagination)
            prior_dist, _ = self.prior(h_t)
            prior_dists.append(prior_dist)

            # 3. Posterior estimation (observation update)
            e_t = self.encoder(observations[:, t])
            post_dist, s_t = self.posterior(h_t, e_t)
            post_dists.append(post_dist)
            s_post_list.append(s_t)

        h_stack = torch.stack(h_list, dim=1)           # (B, T, 128)
        s_stack = torch.stack(s_post_list, dim=1)      # (B, T, 32)

        # Decode observation reconstructions
        recon_obs = self.decoder(h_stack, s_stack)     # (B, T, 49)

        return {
            "reconstructed_obs": recon_obs,
            "deterministic_states": h_stack,
            "stochastic_states": s_stack,
            "posterior_dists": post_dists,
            "prior_dists": prior_dists,
        }

    def initial_posterior_mean(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """
        Deterministic initialization of recurrent state and posterior at timestep t=0.
        
        Args:
            observation: (B, 49) or (49,) Initial normalized observation vector x_0.
            
        Returns:
            Tuple of:
                - h_0: (B, 128) Zero-initialized deterministic hidden state.
                - s_0: (B, 32) Deterministic posterior mean mu_0 = post_dist_0.mean.
                - post_dist_0: Normal distribution object for q(s_0 | h_0, e_0).
        """
        is_unbatched = (observation.ndim == 1)
        if is_unbatched:
            observation = observation.unsqueeze(0)
        B = observation.shape[0]
        device = observation.device

        h_0 = torch.zeros(B, self.cfg.deterministic_dim, device=device, dtype=observation.dtype)
        e_0 = self.encoder(observation)
        post_dist_0, _ = self.posterior(h_0, e_0)
        s_0 = post_dist_0.mean

        if is_unbatched:
            return h_0.squeeze(0), s_0.squeeze(0), post_dist_0
        return h_0, s_0, post_dist_0

    def step_causal_mean(
        self,
        prev_stochastic: torch.Tensor,
        action: torch.Tensor,
        prev_deterministic: torch.Tensor,
        current_observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        """
        Deterministic causal step forward in time from t-1 to t.
        
        Propagates recurrent GRU transition and computes observation posterior using
        the distribution mean (post_dist.mean) rather than stochastic rsample().
        
        Args:
            prev_stochastic: (B, 32) or (32,) Previous stochastic state s_{t-1}.
            action: (B, 8) or (8,) Executed 8-dim action a_{t-1}.
            prev_deterministic: (B, 128) or (128,) Previous deterministic state h_{t-1}.
            current_observation: (B, 49) or (49,) Current normalized observation x_t.
            
        Returns:
            Tuple of:
                - h_t: (B, 128) Updated deterministic hidden state.
                - s_t: (B, 32) Deterministic posterior mean mu_t = post_dist_t.mean.
                - post_dist_t: Normal distribution object for q(s_t | h_t, e_t).
        """
        is_unbatched = (prev_stochastic.ndim == 1)
        if is_unbatched:
            prev_stochastic = prev_stochastic.unsqueeze(0)
            action = action.unsqueeze(0)
            prev_deterministic = prev_deterministic.unsqueeze(0)
            current_observation = current_observation.unsqueeze(0)

        # 1. Deterministic recurrent transition
        h_t = self.recurrent(prev_stochastic, action, prev_deterministic)

        # 2. Observation encoding and posterior evaluation
        e_t = self.encoder(current_observation)
        post_dist_t, _ = self.posterior(h_t, e_t)

        # 3. Deterministic posterior state extraction (mean)
        s_t = post_dist_t.mean

        if is_unbatched:
            return h_t.squeeze(0), s_t.squeeze(0), post_dist_t
        return h_t, s_t, post_dist_t

    def imagine(
        self,
        h_0: torch.Tensor,
        s_0: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Open-loop multi-step latent rollout conditioned solely on actions.
        
        Args:
            h_0: (B, 128) Initial deterministic state
            s_0: (B, 32) Initial stochastic state
            actions: (B, H, 8) Action sequence of horizon H
            
        Returns:
            Tuple of:
                - imagined_obs: (B, H, 49)
                - imagined_h: (B, H, 128)
                - imagined_s: (B, H, 32)
        """
        B, H, _ = actions.shape
        h_t = h_0
        s_t = s_0

        h_list = []
        s_list = []

        for tau in range(H):
            a_tau = actions[:, tau]
            h_t = self.recurrent(s_t, a_tau, h_t)
            prior_dist, s_t = self.prior(h_t)
            # Use distribution mean during evaluation imagination for deterministic evaluation
            s_t = prior_dist.mean
            h_list.append(h_t)
            s_list.append(s_t)

        h_stack = torch.stack(h_list, dim=1)  # (B, H, 128)
        s_stack = torch.stack(s_list, dim=1)  # (B, H, 32)
        imagined_obs = self.decoder(h_stack, s_stack)  # (B, H, 49)

        return imagined_obs, h_stack, s_stack

    @staticmethod
    def quaternion_geodesic_angle(q_true: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
        """Computes true geodesic angle in radians for reporting: 2 * arccos(clamp(|<q, q_hat>|, 0, 1))."""
        q_true_n = q_true / torch.norm(q_true, dim=-1, keepdim=True).clamp(min=1e-8)
        q_pred_n = q_pred / torch.norm(q_pred, dim=-1, keepdim=True).clamp(min=1e-8)
        inner = torch.clamp(torch.abs(torch.sum(q_true_n * q_pred_n, dim=-1)), 0.0, 1.0)
        return torch.mean(2.0 * torch.acos(inner))

    @staticmethod
    def quaternion_geodesic_loss(q_true: torch.Tensor, q_pred: torch.Tensor) -> torch.Tensor:
        """
        Orientation-aware loss with non-singular gradients:
        loss = 1.0 - |<q, q_hat>|
        Monotonic with angular error, strictly 0.0 for q and -q, gradient bounded at -1.0.
        """
        q_true_n = q_true / torch.norm(q_true, dim=-1, keepdim=True).clamp(min=1e-8)
        q_pred_n = q_pred / torch.norm(q_pred, dim=-1, keepdim=True).clamp(min=1e-8)
        inner = torch.clamp(torch.abs(torch.sum(q_true_n * q_pred_n, dim=-1)), 0.0, 1.0)
        return torch.mean(1.0 - inner)

    def compute_loss(
        self,
        targets: torch.Tensor,
        predictions: Dict[str, Any],
        kl_scale: float = 0.10,
        kl_balance: float = 0.80,
        free_nats: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive loss function with weighted reconstruction,
        quaternion geodesic loss, binary contact BCE, and balanced KL divergence.
        """
        recon = predictions["reconstructed_obs"]  # (B, T, 49)
        B, T, D = targets.shape

        # 1. Continuous Reconstruction Loss (Weighted MSE)
        weights = torch.ones(D, device=targets.device)
        if loss_weights:
            # Indices:
            # 0:7 qpos, 7:14 qvel, 14:15 grip, 15:18 ee_pos, 18:22 ee_quat,
            # 22:25 obj_pos, 25:29 obj_quat, 29:32 obj_vel, 32:35 goal,
            # 35:39 obs, 39:45 proxy, 45:46 d_min, 46:47 d_zone, 47:49 contacts
            weights[0:7] = loss_weights.get("robot_joints", 1.0)
            weights[7:14] = loss_weights.get("robot_velocities", 0.5)
            weights[14:15] = loss_weights.get("gripper_aperture", 1.0)
            weights[15:18] = loss_weights.get("ee_position", 1.5)
            weights[22:25] = loss_weights.get("object_position", 2.0)
            weights[29:32] = loss_weights.get("object_velocity", 0.5)
            weights[32:35] = loss_weights.get("goal_position", 1.0)
            weights[35:39] = loss_weights.get("obstacle_state", 1.5)
            weights[39:45] = loss_weights.get("proxy_state", 2.0)
            weights[45:46] = loss_weights.get("clearance_distance", 3.0)
            weights[46:47] = loss_weights.get("zone_distance", 2.5)

        # Exclude quaternions [18:22, 25:29] and contact bits [47:49] from standard MSE
        cont_mask = torch.ones(D, dtype=torch.bool, device=targets.device)
        cont_mask[18:22] = False
        cont_mask[25:29] = False
        cont_mask[47:49] = False

        diff_sq = (recon[..., cont_mask] - targets[..., cont_mask]) ** 2
        w_cont = weights[cont_mask]
        mse_loss = torch.mean(torch.sum(diff_sq * w_cont, dim=-1))

        # 2. Orientation-Aware Geodesic Loss
        ee_q_true = targets[..., 18:22]
        ee_q_pred = recon[..., 18:22]
        ee_rot_loss = self.quaternion_geodesic_loss(ee_q_true, ee_q_pred)
        obj_q_true = targets[..., 25:29]
        obj_q_pred = recon[..., 25:29]
        obj_rot_loss = self.quaternion_geodesic_loss(obj_q_true, obj_q_pred)

        ee_rot_angle = self.quaternion_geodesic_angle(ee_q_true, ee_q_pred)
        obj_rot_angle = self.quaternion_geodesic_angle(obj_q_true, obj_q_pred)

        rot_weight = loss_weights.get("ee_orientation", 2.0) if loss_weights else 2.0
        total_rot_loss = rot_weight * (ee_rot_loss + obj_rot_loss)

        # 3. Binary Contact Loss
        contact_true = targets[..., 47:49]
        contact_pred = recon[..., 47:49]
        contact_loss = F.binary_cross_entropy_with_logits(contact_pred, contact_true)
        contact_weight = loss_weights.get("contact_flags", 2.0) if loss_weights else 2.0

        # Total reconstruction loss
        total_recon = mse_loss + total_rot_loss + contact_weight * contact_loss

        # 4. Balanced KL Divergence (SafeDreamer / Dreamer formulation)
        post_dists = predictions["posterior_dists"]
        prior_dists = predictions["prior_dists"]
        kl_losses = []

        for t in range(1, T):
            p_dist = prior_dists[t]
            q_dist = post_dists[t]

            # Detached prior for representation update
            p_detached = Normal(p_dist.mean.detach(), p_dist.stddev.detach())
            # Detached posterior for dynamics prediction update
            q_detached = Normal(q_dist.mean.detach(), q_dist.stddev.detach())

            kl_rep = kl_divergence(q_dist, p_detached).sum(dim=-1)
            kl_dyn = kl_divergence(q_detached, p_dist).sum(dim=-1)

            kl_val = kl_balance * kl_rep + (1.0 - kl_balance) * kl_dyn
            kl_val = torch.maximum(kl_val, torch.tensor(free_nats, device=targets.device))
            kl_losses.append(kl_val)

        mean_kl = torch.mean(torch.stack(kl_losses))
        total_loss = total_recon + kl_scale * mean_kl

        return {
            "loss": total_loss,
            "recon_loss": total_recon,
            "mse_loss": mse_loss,
            "ee_rot_loss_rad": ee_rot_angle,
            "obj_rot_loss_rad": obj_rot_angle,
            "contact_loss": contact_loss,
            "kl_loss": mean_kl,
        }
