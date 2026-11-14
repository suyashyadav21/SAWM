"""
SAWM World Model Package.
Implements Recurrent State-Space Model (RSSM), trajectory dataset management,
training pipelines, and multi-horizon latent imagination evaluators.
"""

from world_model.rssm import RSSM, RSSMConfig
from world_model.dataset import TrajectorySequenceDataset, TrajectoryNormalizer

__all__ = ["RSSM", "RSSMConfig", "TrajectorySequenceDataset", "TrajectoryNormalizer"]
