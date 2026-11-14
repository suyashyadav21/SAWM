"""
SAWM - Milestone 4 RSSM Training Script.
Trains the Recurrent State-Space Model on collected trajectory dataset.

Features:
- Loads YAML config (`configs/world_model.yaml`)
- Loads train and validation sequence datasets
- Uses CUDA GPU if available (RTX 3050 Ti) with automatic CPU fallback
- Runs full training with KL balancing and feature-weighted losses
- Saves best checkpoint to `checkpoints/rssm_best.pt`

Usage:
  python scripts/train_world_model.py
  python scripts/train_world_model.py --epochs 30 --lr 0.0003
"""

import os
import sys
import glob
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import TrajectorySequenceDataset, TrajectoryNormalizer
from world_model.trainer import RSSMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="SAWM Milestone 4 - RSSM Training Script")
    parser.add_argument("--config", type=str, default="configs/world_model.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size

    # Device selection
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print("=" * 80)
    print("SAWM - Milestone 4: Training Predictive World Model (RSSM)")
    print("=" * 80)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs:           {config['training']['epochs']}")
    print(f"Batch Size:       {config['dataset']['batch_size']}")
    print(f"Learning Rate:    {config['training']['learning_rate']}")
    print(f"KL Scale (beta):  {config['training']['kl_scale']}")
    print(f"KL Balance (alp): {config['training']['kl_balance']}")
    print("=" * 80)

    # Load normalizer
    norm_path = os.path.join(config["training"]["checkpoint_dir"], config["training"]["normalizer_name"])
    normalizer = TrajectoryNormalizer(obs_dim=config["model"]["obs_dim"])
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
        print(f"[INFO] Loaded training normalizer from {norm_path}")
    else:
        print("[WARNING] Normalizer checkpoint not found; operating without feature standardization.")

    # Locate trajectory files
    data_dir = config["dataset"]["data_dir"]
    train_files = sorted(glob.glob(os.path.join(data_dir, "train", "*.npz")))
    val_files = sorted(glob.glob(os.path.join(data_dir, "val", "*.npz")))

    print(f"[INFO] Located {len(train_files)} training episodes and {len(val_files)} validation episodes.")
    assert len(train_files) > 0, f"No train files found in {os.path.join(data_dir, 'train')}"
    assert len(val_files) > 0, f"No val files found in {os.path.join(data_dir, 'val')}"

    # Build sequence datasets (strictly respecting episode boundaries)
    seq_len = config["dataset"]["sequence_length"]
    train_dataset = TrajectorySequenceDataset(train_files, sequence_length=seq_len, normalizer=normalizer)
    val_dataset = TrajectorySequenceDataset(val_files, sequence_length=seq_len, normalizer=normalizer)

    print(f"[INFO] Extracted {len(train_dataset)} train sequences and {len(val_dataset)} val sequences (L={seq_len}).")

    batch_size = config["dataset"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize RSSM
    m_cfg = config["model"]
    rssm_cfg = RSSMConfig(
        obs_dim=m_cfg["obs_dim"],
        action_dim=m_cfg["action_dim"],
        encoder_hidden_dim=m_cfg["encoder_hidden_dim"],
        deterministic_dim=m_cfg["deterministic_dim"],
        stochastic_dim=m_cfg["stochastic_dim"],
        prior_hidden_dim=m_cfg["prior_hidden_dim"],
        posterior_hidden_dim=m_cfg["posterior_hidden_dim"],
        decoder_hidden_dim=m_cfg["decoder_hidden_dim"],
        min_std=m_cfg["min_std"],
        max_std=m_cfg["max_std"],
    )
    model = RSSM(rssm_cfg)

    # Initialize trainer and train
    trainer = RSSMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    history = trainer.train(verbose=True)

    print("\n[INFO] Training complete. Run scripts/evaluate_world_model.py for multi-horizon benchmarks.")


if __name__ == "__main__":
    main()
