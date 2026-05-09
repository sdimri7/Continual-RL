"""
Custom environment wrappers that extend ManiSkill's built-ins.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import torch
from mani_skill.utils.wrappers import RecordEpisode


class DriveRecordEpisode(RecordEpisode):
    """RecordEpisode that automatically saves to a project-backed output dir.

    On first use it creates the directory if it doesn't exist.
    Uses project root detection to avoid hardcoded paths.

    Usage
    -----
    >>> from utils.project import get_project_root
    >>> env = DriveRecordEpisode(base_env, output_dir=os.path.join(get_project_root(), "eval_videos"))
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str = None,
        max_steps_per_video: int = 200,
        save_trajectory: bool = False,
    ) -> None:
        if output_dir is None:
            from utils.project import get_eval_video_dir
            output_dir = get_eval_video_dir()
        os.makedirs(output_dir, exist_ok=True)
        super().__init__(
            env,
            output_dir=output_dir,
            save_trajectory=save_trajectory,
            max_steps_per_video=max_steps_per_video,
        )

    @property
    def drive_root(self) -> str:
        return self._drive_root


class FrameStackWrapper(gym.Wrapper):
    """Stack the last `n_frames` RGB observations along the channel axis.

    Diffusion Policy typically uses a short observation history (obs_horizon).
    This wrapper pre-processes the stacking so the policy receives a single
    tensor of shape (B, C*n_frames, H, W).

    Args:
        env: Wrapped environment.
        n_frames: Number of frames to stack (matches obs_horizon in config).
        key: Key in obs dict that holds the RGB image tensor.
    """

    def __init__(self, env: gym.Env, n_frames: int = 2, key: str = "rgb") -> None:
        super().__init__(env)
        self.n_frames = n_frames
        self.key = key
        self._buffer: list[torch.Tensor] = []

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        img = self._extract(obs)
        self._buffer = [img] * self.n_frames
        obs[self.key] = self._stack()
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._buffer.pop(0)
        self._buffer.append(self._extract(obs))
        obs[self.key] = self._stack()
        return obs, rew, term, trunc, info

    def _extract(self, obs) -> torch.Tensor:
        # sensor_data -> base_camera -> rgb  OR  direct rgb key
        if "sensor_data" in obs:
            return obs["sensor_data"]["base_camera"]["rgb"]
        return obs[self.key]

    def _stack(self) -> torch.Tensor:
        # (B, H, W, C) * n  ->  (B, H, W, C*n)
        return torch.cat(self._buffer, dim=-1)
