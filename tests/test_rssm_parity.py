"""
Tests for PR-01 RSSM Online/Batch Parity and Action Representation.

Verifies:
1. Exact numerical agreement between causal online posterior-mean updates
   and an independent deterministic batch reference rollout across T=50 timesteps.
2. Exact matching of the 8-dim action representation against the M4 training pipeline.
3. Provenance and property verification of TrajectoryNormalizer.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pytest
import torch

from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import TrajectoryNormalizer, ObservationBuilder


def deterministic_batch_forward(
    model: RSSM,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> Dict[str, Any]:
    """
    Independent batch reference unroll using posterior distribution means at every step.
    
    Args:
        model: Evaluated RSSM model in eval mode.
        observations: (B, T, 49) Sequence of T normalized observations.
        actions: (B, T-1, 8) Sequence of T-1 actions.
        
    Returns:
        Dict with deterministic states h (B, T, 128), stochastic means s (B, T, 32),
        and reconstructed observations (B, T, 49).
    """
    B, T, _ = observations.shape
    device = observations.device

    h_0 = torch.zeros(B, model.cfg.deterministic_dim, device=device, dtype=observations.dtype)
    e_0 = model.encoder(observations[:, 0])
    post_dist_0, _ = model.posterior(h_0, e_0)
    s_0 = post_dist_0.mean

    h_list = [h_0]
    s_list = [s_0]

    h_curr = h_0
    s_curr = s_0

    for t in range(1, T):
        a_prev = actions[:, t - 1]
        h_curr = model.recurrent(s_curr, a_prev, h_curr)
        e_curr = model.encoder(observations[:, t])
        post_dist_curr, _ = model.posterior(h_curr, e_curr)
        s_curr = post_dist_curr.mean

        h_list.append(h_curr)
        s_list.append(s_curr)

    h_stack = torch.stack(h_list, dim=1)  # (B, T, 128)
    s_stack = torch.stack(s_list, dim=1)  # (B, T, 32)
    recon_obs = model.decoder(h_stack, s_stack)

    return {
        "deterministic_states": h_stack,
        "stochastic_states": s_stack,
        "reconstructed_obs": recon_obs,
    }


@pytest.fixture(scope="module")
def loaded_rssm() -> RSSM:
    """Load the trained M4 RSSM checkpoint in eval mode on CPU."""
    ckpt_path = "checkpoints/rssm_best.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
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
    model = RSSM(rssm_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@pytest.fixture(scope="module")
def loaded_normalizer() -> TrajectoryNormalizer:
    """Load the trained M4 TrajectoryNormalizer from checkpoints."""
    norm_path = "checkpoints/normalizer.pt"
    assert os.path.exists(norm_path), f"Normalizer not found: {norm_path}"
    
    data = torch.load(norm_path, map_location="cpu", weights_only=False)
    norm = TrajectoryNormalizer(obs_dim=49)
    norm.mean = data["mean"].numpy() if isinstance(data["mean"], torch.Tensor) else np.array(data["mean"])
    norm.std = data["std"].numpy() if isinstance(data["std"], torch.Tensor) else np.array(data["std"])
    norm.exempt_indices = set(data["exempt_indices"])
    norm.is_fit = bool(data["is_fit"])
    return norm


# =============================================================================
# PR-01 Acceptance Tests: Numerical Parity Across 50 Observations & 49 Actions
# =============================================================================

def test_pr01_rssm_online_batch_parity_synthetic(loaded_rssm: RSSM):
    """
    PR-01: Verify strict numerical agreement between sequential causal steps
    and batch reference unroll over 50 steps (49 transitions) with synthetic inputs.
    Tolerances: max |mu_seq - mu_batch| < 1e-5 and max |h_seq - h_batch| < 1e-5.
    """
    model = loaded_rssm
    torch.manual_seed(42)

    T = 50
    B = 2
    # Generate synthetic normalized observations and standardized actions
    obs_seq = torch.randn(B, T, 49, dtype=torch.float32)
    act_seq = torch.clamp(torch.randn(B, T - 1, 8, dtype=torch.float32), -1.5, 1.5)
    # Ensure gripper action is in [0, 1]
    act_seq[:, :, 7] = torch.rand(B, T - 1)

    with torch.no_grad():
        # 1. Independent batch reference rollout
        batch_out = deterministic_batch_forward(model, obs_seq, act_seq)
        h_batch = batch_out["deterministic_states"]   # (B, T, 128)
        s_batch = batch_out["stochastic_states"]      # (B, T, 32)
        recon_batch = batch_out["reconstructed_obs"]  # (B, T, 49)

        # 2. Sequential online causal step-by-step propagation
        h_seq_list = []
        s_seq_list = []

        # Stage 0: Initial step t=0
        h_0, s_0, post_dist_0 = model.initial_posterior_mean(obs_seq[:, 0])
        h_seq_list.append(h_0)
        s_seq_list.append(s_0)

        # Stage 1: Sequential steps t=1..49
        h_prev = h_0
        s_prev = s_0
        for t in range(1, T):
            a_prev = act_seq[:, t - 1]
            x_t = obs_seq[:, t]
            h_t, s_t, post_dist_t = model.step_causal_mean(s_prev, a_prev, h_prev, x_t)
            h_seq_list.append(h_t)
            s_seq_list.append(s_t)
            h_prev = h_t
            s_prev = s_t

        h_seq = torch.stack(h_seq_list, dim=1)  # (B, T, 128)
        s_seq = torch.stack(s_seq_list, dim=1)  # (B, T, 32)
        recon_seq = model.decoder(h_seq, s_seq)

        # 3. Acceptance Verification
        max_h_err = torch.max(torch.abs(h_seq - h_batch)).item()
        max_s_err = torch.max(torch.abs(s_seq - s_batch)).item()
        max_recon_err = torch.max(torch.abs(recon_seq - recon_batch)).item()

        print(f"\n[PR-01 Synthetic] Max h error: {max_h_err:.2e}, Max s error: {max_s_err:.2e}, Max recon error: {max_recon_err:.2e}")

        # PR-01 Acceptance Thresholds: strictly below 1e-5
        assert max_h_err < 1e-5, f"Deterministic state error {max_h_err} >= 1e-5"
        assert max_s_err < 1e-5, f"Stochastic posterior mean error {max_s_err} >= 1e-5"
        assert max_recon_err < 1e-5, f"Reconstruction error {max_recon_err} >= 1e-5"

        # Explicit check for initial posterior at t=0
        diff_h0 = torch.max(torch.abs(h_seq[:, 0] - h_batch[:, 0])).item()
        diff_s0 = torch.max(torch.abs(s_seq[:, 0] - s_batch[:, 0])).item()
        assert diff_h0 == 0.0, f"Initial hidden state h0 is not zero: {diff_h0}"
        assert diff_s0 < 1e-5, f"Initial posterior mean s0 error {diff_s0} >= 1e-5"


def test_pr01_rssm_online_batch_parity_dataset_episode(loaded_rssm: RSSM, loaded_normalizer: TrajectoryNormalizer):
    """
    PR-01: Verify parity using actual real recorded trajectory observations and actions
    from the processed training dataset.
    """
    model = loaded_rssm
    normalizer = loaded_normalizer

    data_dirs = ["data/trajectories/train", "data/trajectories/val"]
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d) and len([f for f in os.listdir(d) if f.endswith(".npz")]) > 0:
            data_dir = d
            break

    if data_dir is None:
        pytest.skip("No trajectory .npz datasets found in data/trajectories/; skipping real episode parity test.")

    episodes = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    ep_path = os.path.join(data_dir, episodes[0])
    raw_data = np.load(ep_path)
    raw_obs = raw_data["observations"].astype(np.float32)  # (N, 49)
    raw_act = raw_data["actions"].astype(np.float32)       # (N-1, 8)

    assert len(raw_obs) >= 50, f"Episode length {len(raw_obs)} < 50"

    # Take first 50 observations and 49 actions
    obs_slice = raw_obs[:50]
    act_slice = raw_act[:49]

    # Pre-normalize observations with TrajectoryNormalizer
    obs_norm = normalizer.normalize(obs_slice)

    obs_tensor = torch.from_numpy(obs_norm).unsqueeze(0)  # (1, 50, 49)
    act_tensor = torch.from_numpy(act_slice).unsqueeze(0)  # (1, 49, 8)

    with torch.no_grad():
        batch_out = deterministic_batch_forward(model, obs_tensor, act_tensor)
        h_batch = batch_out["deterministic_states"]
        s_batch = batch_out["stochastic_states"]

        h_seq_list = []
        s_seq_list = []

        h_0, s_0, _ = model.initial_posterior_mean(obs_tensor[:, 0])
        h_seq_list.append(h_0)
        s_seq_list.append(s_0)

        h_prev = h_0
        s_prev = s_0
        for t in range(1, 50):
            a_prev = act_tensor[:, t - 1]
            x_t = obs_tensor[:, t]
            h_t, s_t, _ = model.step_causal_mean(s_prev, a_prev, h_prev, x_t)
            h_seq_list.append(h_t)
            s_seq_list.append(s_t)
            h_prev = h_t
            s_prev = s_t

        h_seq = torch.stack(h_seq_list, dim=1)
        s_seq = torch.stack(s_seq_list, dim=1)

        max_h_err = torch.max(torch.abs(h_seq - h_batch)).item()
        max_s_err = torch.max(torch.abs(s_seq - s_batch)).item()

        print(f"\n[PR-01 Real Episode] Max h error: {max_h_err:.2e}, Max s error: {max_s_err:.2e}")
        assert max_h_err < 1e-5, f"Deterministic state error {max_h_err} >= 1e-5 on real data"
        assert max_s_err < 1e-5, f"Stochastic posterior mean error {max_s_err} >= 1e-5 on real data"


# =============================================================================
# Action Representation and Normalization Verification
# =============================================================================

def test_m4_action_encoding_verification():
    """
    Verify that extract_action_vector strictly constructs the exact 8-dim action
    representation expected by the M4 RSSM training pipeline:
    - Indices [0:7]: Arm joint position commands in radians.
    - Index [7]: Normalized gripper command g_norm = clip(1.0 - gripper_cmd / 255.0, 0.0, 1.0).
    """
    # 1. Standard open command (255.0) -> g_norm = 0.0
    q_arm = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
    act_open = ObservationBuilder.extract_action_vector(q_arm, 255.0)
    assert act_open.shape == (8,)
    assert act_open.dtype == np.float32
    np.testing.assert_allclose(act_open[:7], q_arm, atol=1e-6)
    assert act_open[7] == pytest.approx(0.0, abs=1e-6)

    # 2. Standard closed clamp command (0.0) -> g_norm = 1.0
    act_closed = ObservationBuilder.extract_action_vector(q_arm, 0.0)
    assert act_closed.shape == (8,)
    np.testing.assert_allclose(act_closed[:7], q_arm, atol=1e-6)
    assert act_closed[7] == pytest.approx(1.0, abs=1e-6)

    # 3. Intermediate command (127.5) -> g_norm = 0.5
    act_mid = ObservationBuilder.extract_action_vector(q_arm, 127.5)
    assert act_mid[7] == pytest.approx(0.5, abs=1e-4)

    # 4. Out of range clamping (<0 and >255)
    act_clamped_high = ObservationBuilder.extract_action_vector(q_arm, 300.0)
    assert act_clamped_high[7] == pytest.approx(0.0, abs=1e-6)

    act_clamped_low = ObservationBuilder.extract_action_vector(q_arm, -50.0)
    assert act_clamped_low[7] == pytest.approx(1.0, abs=1e-6)

    # 5. Invalid arm target length raises AssertionError
    with pytest.raises(AssertionError):
        ObservationBuilder.extract_action_vector(np.zeros(6), 255.0)


def test_normalizer_provenance_and_exempt_indices(loaded_normalizer: TrajectoryNormalizer):
    """
    Verify provenance and invariant properties of the TrajectoryNormalizer:
    1. Only 49-dim observations are normalized; actions are NOT normalized by TrajectoryNormalizer.
    2. Exempt indices have mean == 0.0 and std == 1.0.
    3. Normalization and denormalization are exact inverses.
    """
    norm = loaded_normalizer
    assert norm.is_fit is True
    assert norm.obs_dim == 49
    assert len(norm.mean) == 49
    assert len(norm.std) == 49

    expected_exempt = {18, 19, 20, 21, 25, 26, 27, 28, 38, 47, 48}
    assert norm.exempt_indices == expected_exempt, f"Exempt indices mismatch: {norm.exempt_indices}"

    # Verify exempt indices have mean=0.0 and std=1.0
    for idx in expected_exempt:
        assert norm.mean[idx] == pytest.approx(0.0, abs=1e-6), f"Exempt index {idx} mean is non-zero: {norm.mean[idx]}"
        assert norm.std[idx] == pytest.approx(1.0, abs=1e-6), f"Exempt index {idx} std is not 1.0: {norm.std[idx]}"

    # Verify invertibility
    test_vec = np.random.uniform(-2.0, 2.0, size=(10, 49)).astype(np.float32)
    normalized = norm.normalize(test_vec)
    reconstructed = norm.denormalize(normalized)
    np.testing.assert_allclose(reconstructed, test_vec, rtol=1e-5, atol=1e-5)
