"""
PyTorch datasets for Diffusion Policy training.

PushTDemoDataset  – loads raw ManiSkill .h5 trajectories and returns
                    (obs_seq, action_seq) windows ready for imitation learning.

DiffusionPolicyDataset – wraps PushTDemoDataset to return the exact
                          (obs_cond, noisy_action, noise, timestep) format
                          expected by the Diffusion Policy training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ── raw trajectory dataset ─────────────────────────────────────────────────
class PushTDemoDataset(Dataset):
    """Sliding-window dataset over Push-T expert demonstrations.

    Each item is a dict with keys:
        "obs"     – float32 tensor (obs_horizon, *obs_shape)
        "action"  – float32 tensor (pred_horizon, action_dim)

    Args:
        traj_path: Path to the converted .h5 file (rgb + pd_ee_delta_pos).
        obs_horizon: Number of past observations used as conditioning (Toy: 2).
        pred_horizon: Number of future actions to predict (Toy: 16).
        obs_key: Key inside each trajectory for observations.
                 Use "obs/agent/qpos" for state, or
                 "obs/sensor_data/base_camera/rgb" for images.
        action_key: Key for actions (default "actions").
        image_size: Resize images to (H, W) if obs_key is an image. None = no resize.
        pad_before: Pad the start of each episode so every timestep can be
                    a valid sample (copies the first frame).
    """

    def __init__(
        self,
        traj_path: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        obs_key: str = "obs/sensor_data/base_camera/rgb",
        action_key: str = "actions",
        image_size: tuple[int, int] | None = (96, 96),
        pad_before: bool = True,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_key = obs_key
        self.action_key = action_key
        self.image_size = image_size

        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._load(traj_path, pad_before)

    # ── loading ────────────────────────────────────────────────────────────
    def _load(self, traj_path: str, pad_before: bool) -> None:
        print(f"[dataset] Loading {traj_path} …")
        with h5py.File(traj_path, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            for tk in traj_keys:
                traj = f[tk]
                obs = self._read_obs(traj)     # (T, *obs_shape)
                actions = traj[self.action_key][:]  # (T, action_dim)
                T = len(actions)

                if pad_before:
                    pad_obs = np.repeat(obs[:1], self.obs_horizon - 1, axis=0)
                    obs = np.concatenate([pad_obs, obs], axis=0)

                # slide window
                for t in range(T - self.pred_horizon + 1):
                    obs_window = obs[t: t + self.obs_horizon]
                    act_window = actions[t: t + self.pred_horizon]
                    self._samples.append((obs_window.copy(), act_window.copy()))

        print(f"[dataset] {len(self._samples)} samples loaded.")

    def _read_obs(self, traj: h5py.Group) -> np.ndarray:
        keys = self.obs_key.split("/")
        node = traj
        for k in keys:
            node = node[k]
        obs = node[:]  # (T, …)

        if self.image_size is not None and obs.ndim == 4:
            obs = self._resize_images(obs)

        obs = obs.astype(np.float32)
        if obs.max() > 1.5:
            obs = obs / 255.0  # normalise uint8 images to [0, 1]

        return obs

    def _resize_images(self, imgs: np.ndarray) -> np.ndarray:
        """Resize a batch of (T, H, W, C) images using cv2 if available."""
        if self.image_size is None:
            return imgs
        try:
            import cv2
        except ImportError:
            return imgs

        target_h, target_w = self.image_size
        T, _H, _W, C = imgs.shape
        out = np.empty((T, target_h, target_w, C), dtype=imgs.dtype)
        for i, img in enumerate(imgs):
            out[i] = cv2.resize(img, (target_w, target_h))
        return out

    # ── dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs, act = self._samples[idx]
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(act),
        }


# ── normalisation statistics ───────────────────────────────────────────────
class Normalizer:
    """Min-max normaliser that maps values to [-1, 1].

    Fit once on the full dataset, then save/load the stats so you don't
    recompute across sessions.

    Args:
        save_path: If given, stats are stored as a .npz file on Drive.
    """

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path
        self.stats: dict[str, np.ndarray] = {}

    def fit(self, dataset: PushTDemoDataset) -> None:
        obs_list, act_list = [], []
        for item in dataset:
            obs_list.append(item["obs"].numpy())
            act_list.append(item["action"].numpy())

        obs_all = np.concatenate(obs_list, axis=0)
        act_all = np.concatenate(act_list, axis=0)

        self.stats = {
            "obs_min": obs_all.min(axis=0),
            "obs_max": obs_all.max(axis=0),
            "act_min": act_all.min(axis=0),
            "act_max": act_all.max(axis=0),
        }
        if self.save_path:
            np.savez(self.save_path, **self.stats)
            print(f"[dataset] Normaliser stats saved to {self.save_path}.")

    def load(self, path: str | None = None) -> None:
        path = path or self.save_path
        data = np.load(path)
        self.stats = {k: data[k] for k in data.files}
        print(f"[dataset] Normaliser stats loaded from {path}.")

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(self.stats["act_min"], dtype=x.dtype, device=x.device)
        hi = torch.tensor(self.stats["act_max"], dtype=x.dtype, device=x.device)
        return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0

    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(self.stats["act_min"], dtype=x.dtype, device=x.device)
        hi = torch.tensor(self.stats["act_max"], dtype=x.dtype, device=x.device)
        return (x + 1.0) / 2.0 * (hi - lo + 1e-8) + lo


# ── diffusion policy dataset ───────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """Wraps PushTDemoDataset and applies action normalisation.

    Returns dicts with:
        "obs"    – (obs_horizon, *obs_shape) float32
        "action" – (pred_horizon, action_dim) float32, normalised to [-1, 1]
    """

    def __init__(
        self,
        base_dataset: PushTDemoDataset,
        normalizer: Normalizer,
    ) -> None:
        self.base = base_dataset
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]
        item["action"] = self.normalizer.normalize_action(item["action"])
        return item
