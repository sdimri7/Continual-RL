"""
PyTorch datasets for Diffusion Policy training.

PushTDemoDataset  – loads raw ManiSkill .h5 trajectories and returns
                    (obs_seq, state_seq, action_seq) windows ready for
                    imitation learning.

DiffusionPolicyDataset – thin wrapper that passes items through unchanged.
                          Actions stay in their native [-1, 1] range (the
                          pd_ee_delta_pos action space guarantees this), so
                          no normaliser is applied.
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
        "obs"    – float32 tensor (obs_horizon, H, W, C) in [0, 1]
        "state"  – float32 tensor (obs_horizon, state_dim)  concatenated
                   proprioceptive state from obs/agent/* and obs/extra/*
        "action" – float32 tensor (pred_horizon, action_dim) in [-1, 1]

    Args:
        traj_path:   Path to the converted .h5 file (rgb + pd_ee_delta_pos).
        obs_horizon: Number of past observations used as conditioning (default: 2).
        pred_horizon: Number of future actions to predict (default: 16).
        obs_key:     HDF5 key for image observations.
        action_key:  HDF5 key for actions.
        image_size:  Resize images to (H, W). None = no resize.
        pad_before:  Pad the start of each episode by repeating the first frame.
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

        # Each sample is (obs_window, state_window, act_window)
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._load(traj_path, pad_before)

    # ── loading ────────────────────────────────────────────────────────────
    def _load(self, traj_path: str, pad_before: bool) -> None:
        print(f"[dataset] Loading {traj_path} …")
        with h5py.File(traj_path, "r") as f:
            traj_keys = [k for k in f.keys() if k.startswith("traj_")]
            for tk in traj_keys:
                traj = f[tk]
                obs = self._read_obs(traj)       # (T or T+1, H, W, C)
                state = self._read_state(traj)   # (T or T+1, state_dim) or None
                actions = traj[self.action_key][:]  # (T, action_dim)
                T = len(actions)

                if obs is None or len(obs) == 0:
                    action_dim = actions.shape[-1]
                    obs = np.zeros((T, action_dim), dtype=np.float32)
                    print(f"  [dataset] No obs in HDF5, using zeros (action_dim={action_dim})")

                if state is None:
                    # Fallback: empty state so the pipeline stays consistent
                    state = np.zeros((len(obs), 0), dtype=np.float32)

                if pad_before:
                    pad_obs = np.repeat(obs[:1], self.obs_horizon - 1, axis=0)
                    obs = np.concatenate([pad_obs, obs], axis=0)

                    pad_state = np.repeat(state[:1], self.obs_horizon - 1, axis=0)
                    state = np.concatenate([pad_state, state], axis=0)

                for t in range(T - self.pred_horizon + 1):
                    obs_window = obs[t: t + self.obs_horizon]
                    state_window = state[t: t + self.obs_horizon]
                    act_window = actions[t: t + self.pred_horizon]
                    self._samples.append(
                        (obs_window.copy(), state_window.copy(), act_window.copy())
                    )

        print(f"[dataset] {len(self._samples)} samples loaded.")
        if self._samples:
            _state_dim = self._samples[0][1].shape[-1]
            print(f"[dataset] state_dim={_state_dim}")

    def _read_obs(self, traj: h5py.Group) -> np.ndarray | None:
        """Read image observations; returns (T, H, W, C) float32 in [0, 1]."""
        try:
            keys = self.obs_key.split("/")
            node = traj
            for k in keys:
                node = node[k]
            obs = node[:]

            if self.image_size is not None and obs.ndim == 4:
                obs = self._resize_images(obs)

            obs = obs.astype(np.float32)
            if obs.max() > 1.5:
                obs = obs / 255.0
            return obs
        except (KeyError, TypeError):
            return None

    def _read_state(self, traj: h5py.Group) -> np.ndarray | None:
        """Concatenate all leaf arrays under obs/agent and obs/extra.

        Mirrors the official ManiSkill baseline:
            list(obs["agent"].values()) + list(obs["extra"].values())
        """
        parts: list[np.ndarray] = []
        for group_path in ("obs/agent", "obs/extra"):
            keys = group_path.split("/")
            try:
                node = traj
                for k in keys:
                    node = node[k]
                parts.extend(self._collect_leaves(node))
            except (KeyError, TypeError):
                pass

        if not parts:
            return None

        # All parts share the same T (or T+1) leading dimension
        # Flatten each to (T, -1) and concatenate along dim 1
        flat = []
        for arr in parts:
            arr = arr.astype(np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            flat.append(arr)

        return np.concatenate(flat, axis=-1)

    @staticmethod
    def _collect_leaves(node: h5py.Group | h5py.Dataset) -> list[np.ndarray]:
        """Recursively collect all Dataset arrays under an HDF5 group."""
        if isinstance(node, h5py.Dataset):
            return [node[:]]
        arrays: list[np.ndarray] = []
        for key in node.keys():
            arrays.extend(PushTDemoDataset._collect_leaves(node[key]))
        return arrays

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
        obs, state, act = self._samples[idx]
        return {
            "obs": torch.from_numpy(obs),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(act),
        }

    @property
    def state_dim(self) -> int:
        """Dimensionality of the state vector (0 if no state available)."""
        if not self._samples:
            return 0
        return self._samples[0][1].shape[-1]


# ── normalisation statistics ───────────────────────────────────────────────
class Normalizer:
    """Kept for backward compatibility. Not used for action normalisation.

    The pd_ee_delta_pos action space is already bounded to [-1, 1], matching
    the DDPM scheduler's clip_sample=True assumption. No scaling is needed.
    """

    def __init__(self, save_path: str | None = None) -> None:
        self.save_path = save_path
        self.stats: dict[str, np.ndarray] = {}

    def fit(self, dataset: PushTDemoDataset) -> None:
        pass  # no-op: actions stay in native [-1, 1]

    def load(self, path: str | None = None) -> None:
        pass  # no-op

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity

    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity


# ── diffusion policy dataset ───────────────────────────────────────────────
class DiffusionPolicyDataset(Dataset):
    """Thin pass-through wrapper around PushTDemoDataset.

    Actions are NOT normalised — they are already in [-1, 1] for the
    pd_ee_delta_pos controller and clip_sample=True in the DDPM scheduler
    enforces this range at inference time.

    Returns dicts with keys: "obs", "state", "action".
    """

    def __init__(
        self,
        base_dataset: PushTDemoDataset,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.base[idx]

    @property
    def state_dim(self) -> int:
        return self.base.state_dim
