"""
Trajectory Dataset & Observation Vector Builder for SAWM RSSM.

Manages:
- 49-dim structured vector observation assembly
- Quaternion canonicalization (w >= 0, unit sphere projection)
- MuJoCo object velocity extraction and proxy finite differences
- Train/Validation episode-level isolation (zero leakage)
- Feature-wise normalization computed strictly on training episodes
- Windowed sequence dataset with strict episode boundary protection
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import mujoco
from typing import Dict, List, Tuple, Optional, Any


def canonicalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Project quaternion to unit length and enforce sign convention w >= 0."""
    q = np.array(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        q = q / norm
    else:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if q[0] < 0.0:
        q = -q
    return q


class ObservationBuilder:
    """Extracts standardized 49-dim vector observations from SawmEnvironment."""

    def __init__(self, env):
        self.env = env
        self.cube_id = env.target_object_body_id
        self.proxy_id = env.obstacle_manager.proxy_body_id
        self._prev_proxy_pos = None
        self._vel_scratch = np.zeros(6, dtype=np.float64)

    def reset(self):
        """Reset internal history for finite differences."""
        self._prev_proxy_pos = None

    def extract_observation_vector(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """
        Build exact 49-dimensional vector from simulation state:
        [0:7]   Robot arm joint positions (7)
        [7:14]  Robot arm joint velocities (7)
        [14:15] Gripper finger aperture (1)
        [15:18] End-effector Cartesian position (3)
        [18:22] End-effector canonicalized quaternion (4)
        [22:25] Target cube position (3)
        [25:29] Target cube canonicalized quaternion (4)
        [29:32] Target cube world linear velocity (3)
        [32:35] Goal site position (3)
        [35:38] Static obstacle position (3)
        [38:39] Static obstacle active flag (1)
        [39:42] Dynamic proxy position (3)
        [42:45] Dynamic proxy linear velocity (3)
        [45:46] Minimum geometric clearance d_min (1)
        [46:47] Minimum restricted zone distance d_zone (1)
        [47:48] Robot-object contact flag (1)
        [48:49] Robot-hazard collision flag (1)
        """
        robot = raw_obs["robot"]
        qpos = np.array(robot["joint_positions"][:7], dtype=np.float32)
        qvel = np.array(robot["joint_velocities"][:7], dtype=np.float32)
        gripper_w = np.array([float(robot["gripper_position"])], dtype=np.float32)
        ee_pos = np.array(robot["ee_position"], dtype=np.float32)
        ee_quat = canonicalize_quaternion(robot["ee_quaternion"])

        # Target object pose
        obj = raw_obs["target_object"]
        obj_pos = np.array(obj["position"], dtype=np.float32)
        obj_quat = canonicalize_quaternion(obj["quaternion"])

        # World linear velocity via MuJoCo C API (res[3:6])
        mujoco.mj_objectVelocity(self.env.model, self.env.data, mujoco.mjtObj.mjOBJ_BODY, self.cube_id, self._vel_scratch, 0)
        obj_vel = np.array(self._vel_scratch[3:6], dtype=np.float32)

        # Goal position
        goal_pos = np.array(raw_obs["goal"]["position"], dtype=np.float32)

        # Hazards
        haz = raw_obs["hazards"]
        obs_pos = np.array(haz["static_pos"], dtype=np.float32)
        obs_act = np.array([1.0 if haz["static_active"] else 0.0], dtype=np.float32)

        proxy_pos = np.array(haz["proxy_pos"], dtype=np.float32)
        dt = float(self.env.dt * self.env.frame_skip)
        if self._prev_proxy_pos is None or not haz["proxy_active"]:
            proxy_vel = np.zeros(3, dtype=np.float32)
        else:
            proxy_vel = np.array((proxy_pos - self._prev_proxy_pos) / max(1e-6, dt), dtype=np.float32)
        self._prev_proxy_pos = proxy_pos.copy()

        # Distance & safety metrics
        d_metrics = raw_obs.get("distances", None)
        if d_metrics and hasattr(d_metrics, "geometric_min_distance"):
            d_min = float(d_metrics.geometric_min_distance)
            d_min = 1.0 if d_min > 1.0 or np.isnan(d_min) else max(0.0, d_min)
        else:
            d_min = 1.0
        d_min_arr = np.array([d_min], dtype=np.float32)

        zone_states = raw_obs.get("zone_states", [])
        if zone_states:
            d_zone = min([float(z.min_distance_to_zone) for z in zone_states])
            d_zone = 1.0 if d_zone > 1.0 or np.isnan(d_zone) else max(0.0, d_zone)
        else:
            d_zone = 1.0
        d_zone_arr = np.array([d_zone], dtype=np.float32)

        # Contacts
        contacts = raw_obs["contacts"]
        c_obj = np.array([1.0 if contacts.get("robot_object_contact", False) else 0.0], dtype=np.float32)
        c_haz = np.array([1.0 if contacts.get("robot_obstacle_collision", False) or contacts.get("robot_proxy_collision", False) else 0.0], dtype=np.float32)

        vec = np.concatenate([
            qpos,           # 7
            qvel,           # 7
            gripper_w,      # 1
            ee_pos,         # 3
            ee_quat,        # 4
            obj_pos,        # 3
            obj_quat,       # 4
            obj_vel,        # 3
            goal_pos,       # 3
            obs_pos,        # 3
            obs_act,        # 1
            proxy_pos,      # 3
            proxy_vel,      # 3
            d_min_arr,      # 1
            d_zone_arr,     # 1
            c_obj,          # 1
            c_haz,          # 1
        ])
        assert len(vec) == 49, f"Expected 49-dim observation, got {len(vec)}"
        return vec.astype(np.float32)

    @staticmethod
    def extract_action_vector(arm_targets: np.ndarray, gripper_target: float) -> np.ndarray:
        """Extract standardized 8-dim action vector."""
        q_cmd = np.array(arm_targets[:7], dtype=np.float32)
        # Normalize gripper command: 255 -> 0.0 (open), 0 -> 1.0 (closed clamp)
        g_cmd = np.array([np.clip(1.0 - (float(gripper_target) / 255.0), 0.0, 1.0)], dtype=np.float32)
        act = np.concatenate([q_cmd, g_cmd])
        assert len(act) == 8, f"Expected 8-dim action, got {len(act)}"
        return act.astype(np.float32)


class TrajectoryNormalizer:
    """Feature-wise normalization statistics computed strictly from train episodes."""

    def __init__(self, obs_dim: int = 49):
        self.obs_dim = obs_dim
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.std = np.ones(obs_dim, dtype=np.float32)
        self.is_fit = False

        # Indices to exempt from standardization (preserve unit sphere and binary logits)
        # Quaternions: [18:22] (EE quat), [25:29] (Obj quat)
        # Active flag: [38]
        # Contact bits: [47, 48]
        self.exempt_indices = set(list(range(18, 22)) + list(range(25, 29)) + [38, 47, 48])

    def fit(self, train_observations: np.ndarray):
        """Compute mean and std on flattened training observations."""
        flat = train_observations.reshape(-1, self.obs_dim)
        self.mean = np.mean(flat, axis=0).astype(np.float32)
        self.std = np.std(flat, axis=0).astype(np.float32)

        # Protect zero or tiny variance
        self.std[self.std < 1e-4] = 1.0

        # Exempt specified features
        for idx in self.exempt_indices:
            self.mean[idx] = 0.0
            self.std[idx] = 1.0

        self.is_fit = True

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            return x
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            return x
        return x * self.std + self.mean

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
            "exempt_indices": list(self.exempt_indices),
            "is_fit": self.is_fit,
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        self.mean = data["mean"].numpy()
        self.std = data["std"].numpy()
        self.exempt_indices = set(data["exempt_indices"])
        self.is_fit = data["is_fit"]


class TrajectorySequenceDataset(Dataset):
    """
    Sequence chunk dataset with strict episode boundary protection.
    Pre-normalizes and pre-caches episode tensors in RAM for maximum throughput.
    """

    def __init__(
        self,
        file_paths: List[str],
        sequence_length: int = 50,
        normalizer: Optional[TrajectoryNormalizer] = None,
        step_stride: int = 4,
    ):
        self.file_paths = file_paths
        self.seq_len = sequence_length
        self.normalizer = normalizer
        self.step_stride = step_stride

        self.episodes = []
        self.window_indices = []  # List of (ep_idx, start_t)

        for ep_idx, fpath in enumerate(file_paths):
            data = np.load(fpath, allow_pickle=True)
            obs = data["observations"].astype(np.float32)
            act = data["actions"].astype(np.float32)

            ep_len = len(obs)
            assert len(act) == ep_len - 1, f"Alignment mismatch in {fpath}: {len(obs)} obs vs {len(act)} acts"

            # Pre-normalize observation sequence once
            if self.normalizer and self.normalizer.is_fit:
                obs_norm = self.normalizer.normalize(obs)
            else:
                obs_norm = obs

            self.episodes.append({
                "obs": torch.from_numpy(obs_norm),
                "act": torch.from_numpy(act),
                "path": fpath,
                "length": ep_len,
            })

            # Valid start indices guaranteeing window [start, start + seq_len] stays within this single episode
            num_windows = ep_len - self.seq_len
            if num_windows > 0:
                for start_t in range(0, num_windows, self.step_stride):
                    self.window_indices.append((ep_idx, start_t))

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start_t = self.window_indices[idx]
        ep = self.episodes[ep_idx]

        return {
            "observations": ep["obs"][start_t : start_t + self.seq_len],
            "actions": ep["act"][start_t : start_t + self.seq_len - 1],
            "next_observations": ep["obs"][start_t + 1 : start_t + self.seq_len],
            "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
            "start_step": torch.tensor(start_t, dtype=torch.long),
        }
