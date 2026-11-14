"""
Automated Test Suite for Milestone 4: Predictive World Model (RSSM).

Tests:
1. Exact RSSM parameter count (199,089) and submodules.
2. Quaternion canonicalization, unit normalization, and geodesic loss invariance to sign.
3. Dataset integrity, transition alignment (x_t, a_t, x_{t+1}), and train/val zero leakage.
4. RSSM forward pass, loss computation, and finite backward gradients.
5. Open-loop latent multi-step imagination and quaternion projection.
6. MuJoCo object velocity extraction against finite differences.
7. Model checkpoint save and restore verification.

Usage:
  python tests/test_world_model.py
"""

import os
import sys
import glob
import numpy as np
import torch
import mujoco

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SawmEnvironment
from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import ObservationBuilder, TrajectoryNormalizer, TrajectorySequenceDataset, canonicalize_quaternion


def test_rssm_parameter_count_and_submodules():
    """Verify exact submodule and total parameter count (199,089 weights)."""
    print("\n--- Test 1: RSSM Parameter Count & Architecture Breakdown ---")
    cfg = RSSMConfig()
    model = RSSM(cfg)

    param_counts = {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()}
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Encoder:     {param_counts['encoder']:6d} params (Expected: 23,424)")
    print(f"  Recurrent:   {param_counts['recurrent']:6d} params (Expected: 65,280)")
    print(f"  Posterior:   {param_counts['posterior']:6d} params (Expected: 41,408)")
    print(f"  Prior:       {param_counts['prior']:6d} params (Expected: 25,024)")
    print(f"  Decoder:     {param_counts['decoder']:6d} params (Expected: 43,953)")
    print(f"  TOTAL:       {total_params:6d} params (Expected: 199,089)")

    assert param_counts["encoder"] == 23424, f"Encoder param mismatch: {param_counts['encoder']}"
    assert param_counts["recurrent"] == 65280, f"Recurrent param mismatch: {param_counts['recurrent']}"
    assert param_counts["posterior"] == 41408, f"Posterior param mismatch: {param_counts['posterior']}"
    assert param_counts["prior"] == 25024, f"Prior param mismatch: {param_counts['prior']}"
    assert param_counts["decoder"] == 43953, f"Decoder param mismatch: {param_counts['decoder']}"
    assert total_params == 199089, f"Total param mismatch: {total_params}"

    print("  [PASS] Exact parameter count verified.")


def test_quaternion_canonicalization_and_geodesic_loss():
    """Verify quaternion sign invariance, unit normalization, and geodesic loss."""
    print("\n--- Test 2: Quaternion Canonicalization & Geodesic Loss ---")
    # Test canonicalization: w must be >= 0
    q_neg = np.array([-0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
    q_canon = canonicalize_quaternion(q_neg)
    assert q_canon[0] >= 0.0, "Canonicalized w component must be non-negative"
    assert abs(np.linalg.norm(q_canon) - 1.0) < 1e-6, "Canonicalized quaternion must have unit norm"

    # Test geodesic loss invariance to q and -q
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    q2_pos = q1.clone()
    q2_neg = -q1.clone()

    loss_pos = RSSM.quaternion_geodesic_loss(q1, q2_pos)
    loss_neg = RSSM.quaternion_geodesic_loss(q1, q2_neg)

    print(f"  Loss for identical q: {float(loss_pos):.6f} rad")
    print(f"  Loss for antipodal -q: {float(loss_neg):.6f} rad")
    assert abs(float(loss_pos)) < 1e-5, f"Identical quaternions must give 0 loss, got {loss_pos}"
    assert abs(float(loss_neg)) < 1e-5, f"Antipodal quaternions must give 0 loss, got {loss_neg}"

    # Test 90-degree orthogonal rotation gives pi/2 (1.5708 rad) for angle
    q_orth = torch.tensor([0.70710678, 0.70710678, 0.0, 0.0], dtype=torch.float32)
    angle_orth = RSSM.quaternion_geodesic_angle(q1, q_orth)
    loss_orth = RSSM.quaternion_geodesic_loss(q1, q_orth)
    print(f"  Angle for 90-deg rotation: {float(angle_orth):.4f} rad (Expected: ~1.5708 rad)")
    print(f"  Loss for 90-deg rotation:  {float(loss_orth):.4f} (Expected: ~0.2929)")
    assert abs(float(angle_orth) - (np.pi / 2.0)) < 0.05, f"Expected pi/2, got {angle_orth}"
    assert abs(float(loss_orth) - (1.0 - 0.70710678)) < 0.01, f"Expected 0.2929, got {loss_orth}"

    print("  [PASS] Quaternion canonicalization and geodesic loss verified.")


def test_dataset_alignment_and_leakage_protection():
    """Verify trajectory alignment (x_t, a_t, x_{t+1}) and strict train/val separation."""
    print("\n--- Test 3: Dataset Alignment & Leakage Protection ---")
    data_dir = "data/trajectories"
    train_files = sorted(glob.glob(os.path.join(data_dir, "train", "*.npz")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "val", "*.npz")))

    assert len(train_files) >= 10, f"Expected train files, got {len(train_files)}"
    assert len(val_files) >= 5, f"Expected val files, got {len(val_files)}"

    train_ids = set()
    for f in train_files:
        d = np.load(f, allow_pickle=True)
        obs = d["observations"]
        act = d["actions"]
        assert len(obs) == len(act) + 1, f"Alignment error in {f}: {len(obs)} obs vs {len(act)} acts"
        assert obs.shape[1] == 49, f"Obs shape error: {obs.shape}"
        assert act.shape[1] == 8, f"Action shape error: {act.shape}"
        meta = d["metadata"].item() if hasattr(d["metadata"], "item") else d["metadata"]
        train_ids.add(meta["episode_id"])

    val_ids = set()
    for f in val_files:
        d = np.load(f, allow_pickle=True)
        meta = d["metadata"].item() if hasattr(d["metadata"], "item") else d["metadata"]
        val_ids.add(meta["episode_id"])

    # Zero leakage check
    overlap = train_ids.intersection(val_ids)
    print(f"  Train Episode IDs: {len(train_ids)} unique")
    print(f"  Val Episode IDs:   {len(val_ids)} unique")
    print(f"  ID Overlap:        {len(overlap)}")
    assert len(overlap) == 0, f"DATASET LEAKAGE DETECTED! Episodes in both train and val: {overlap}"

    # Test sequence dataset boundary protection
    ds = TrajectorySequenceDataset(train_files[:5], sequence_length=50)
    for i in range(min(10, len(ds))):
        item = ds[i]
        assert item["observations"].shape == (50, 49)
        assert item["actions"].shape == (49, 8)
        assert item["next_observations"].shape == (49, 49)

    print("  [PASS] Dataset alignment and leakage protection verified.")


def test_rssm_forward_pass_and_loss_gradients():
    """Verify forward pass shapes, balanced KL, and finite backward gradients."""
    print("\n--- Test 4: RSSM Forward Pass & Backward Gradients ---")
    model = RSSM()
    B, T = 4, 15
    obs = torch.randn(B, T, 49)
    act = torch.randn(B, T - 1, 8)

    # Normalize quaternions in synthetic test obs
    obs[..., 18:22] = obs[..., 18:22] / torch.norm(obs[..., 18:22], dim=-1, keepdim=True)
    obs[..., 25:29] = obs[..., 25:29] / torch.norm(obs[..., 25:29], dim=-1, keepdim=True)

    out = model(obs, act)
    assert out["reconstructed_obs"].shape == (B, T, 49)
    assert out["deterministic_states"].shape == (B, T, 128)
    assert out["stochastic_states"].shape == (B, T, 32)
    assert len(out["posterior_dists"]) == T
    assert len(out["prior_dists"]) == T

    losses = model.compute_loss(obs, out)
    assert not torch.isnan(losses["loss"]), "Loss is NaN"
    assert not torch.isinf(losses["loss"]), "Loss is Inf"

    # Backward pass
    losses["loss"].backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print(f"  Loss: {float(losses['loss']):.4f} | Recon: {float(losses['recon_loss']):.4f} | KL: {float(losses['kl_loss']):.4f}")
    print("  [PASS] Forward pass and backward gradients verified.")


def test_open_loop_latent_imagination():
    """Verify multi-step latent rollout without ground-truth observations."""
    print("\n--- Test 5: Open-Loop Latent Imagination ---")
    model = RSSM()
    B, H = 2, 20
    h_0, s_0 = model.initial_state(B, torch.device("cpu"))
    actions = torch.randn(B, H, 8)

    imagined_obs, h_stack, s_stack = model.imagine(h_0, s_0, actions)
    assert imagined_obs.shape == (B, H, 49)
    assert h_stack.shape == (B, H, 128)
    assert s_stack.shape == (B, H, 32)

    # Verify predicted quaternions on unit sphere
    ee_quat = imagined_obs[..., 18:22]
    ee_norm = torch.norm(ee_quat, dim=-1)
    assert torch.allclose(ee_norm, torch.ones_like(ee_norm), atol=1e-5), "EE quats not normalized"

    obj_quat = imagined_obs[..., 25:29]
    obj_norm = torch.norm(obj_quat, dim=-1)
    assert torch.allclose(obj_norm, torch.ones_like(obj_norm), atol=1e-5), "Object quats not normalized"

    print(f"  Imagined {H} steps: obs={list(imagined_obs.shape)}, h={list(h_stack.shape)}, s={list(s_stack.shape)}")
    print("  [PASS] Open-loop latent imagination verified.")


def test_velocity_extraction_accuracy():
    """Verify MuJoCo C API velocity extraction against numerical finite difference."""
    print("\n--- Test 6: Velocity Extraction Validation ---")
    env = SawmEnvironment()
    builder = ObservationBuilder(env)
    obs = env.reset()

    # Step physics with arm motion
    p0 = env.data.xpos[env.target_object_body_id].copy()
    for _ in range(5):
        raw_obs = env.step()

    p1 = env.data.xpos[env.target_object_body_id].copy()
    dt = float(env.dt * env.frame_skip)
    fd_vel = (p1 - p0) / (5.0 * dt)

    v_scratch = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.target_object_body_id, v_scratch, 0)
    c_api_vel = v_scratch[3:6]

    print(f"  Finite Difference Vel: {fd_vel}")
    print(f"  MuJoCo C API Vel:      {c_api_vel}")
    assert np.allclose(fd_vel, c_api_vel, atol=0.01), f"Velocity discrepancy: FD={fd_vel} vs C_API={c_api_vel}"
    print("  [PASS] Object velocity extraction verified against finite difference.")
    env.close()


def test_checkpoint_save_and_load():
    """Verify model state saving and restoration."""
    print("\n--- Test 7: Checkpoint Save and Restore ---")
    model1 = RSSM()
    ckpt_path = "checkpoints/test_checkpoint.pt"
    os.makedirs("checkpoints", exist_ok=True)

    torch.save({"model_state_dict": model1.state_dict()}, ckpt_path)

    model2 = RSSM()
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(data["model_state_dict"])

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Loaded parameter does not match saved parameter"

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("  [PASS] Checkpoint save and restore verified.")


if __name__ == "__main__":
    print("=" * 80)
    print("SAWM - Milestone 4 Automated Test Suite (Predictive World Model / RSSM)")
    print("=" * 80)
    try:
        test_rssm_parameter_count_and_submodules()
        test_quaternion_canonicalization_and_geodesic_loss()
        test_dataset_alignment_and_leakage_protection()
        test_rssm_forward_pass_and_loss_gradients()
        test_open_loop_latent_imagination()
        test_velocity_extraction_accuracy()
        test_checkpoint_save_and_load()
        print("\n" + "=" * 80)
        print("ALL MILESTONE 4 UNIT TESTS PASSED SUCCESSFULLY! (7/7)")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n[FAIL] Test assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
