"""
Diffusion Policy: wraps the UNet noise predictor with a DDPM scheduler.

Reference: Chi et al. 2023, https://arxiv.org/abs/2303.04137

Checkpoints are managed via the training config paths, which are set
relative to the project root directory.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .networks import ConditionalUNet1D, ObservationEncoder


class DiffusionPolicy(nn.Module):
    """End-to-end Diffusion Policy model.

    Components
    ----------
    * ``obs_encoder``  – maps observation history to a conditioning vector.
    * ``noise_pred``   – conditional 1-D U-Net that predicts ε given
                         (noisy_action, timestep, obs_cond).
    * ``scheduler``    – DDPM noise scheduler (train) / denoising schedule (eval).

    Args:
        obs_mode: "image" or "state".
        obs_horizon: History length fed as conditioning.
        obs_shape: Shape of a single observation (H, W, C) or (state_dim,).
        action_dim: Dimensionality of a single action vector.
        pred_horizon: Number of future action steps to predict.
        action_horizon: Number of predicted steps to actually execute.
        obs_cond_dim: Latent size of the observation encoder output.
        num_diffusion_steps: Total DDPM timesteps (T in the paper).
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        obs_mode: str = "image",
        obs_horizon: int = 2,
        obs_shape: tuple[int, ...] = (96, 96, 3),
        action_dim: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        obs_cond_dim: int = 256,
        num_diffusion_steps: int = 100,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.obs_mode = obs_mode
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.device = device

        self.obs_encoder = ObservationEncoder(
            obs_mode=obs_mode,
            obs_horizon=obs_horizon,
            obs_shape=obs_shape,
            out_dim=obs_cond_dim,
        )
        self.noise_pred = ConditionalUNet1D(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            obs_cond_dim=obs_cond_dim,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.to(device)

    # ── training step ──────────────────────────────────────────────────────
    def compute_loss(
        self,
        obs: torch.Tensor,    # (B, obs_horizon, *obs_shape)
        action: torch.Tensor, # (B, pred_horizon, action_dim) – normalised
    ) -> torch.Tensor:
        """DDPM training loss: predict ε from the noisy action at a random t."""
        B = obs.shape[0]
        obs = obs.to(self.device)
        action = action.to(self.device)

        obs_cond = self.obs_encoder(obs)

        noise = torch.randn_like(action)
        t = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        )
        noisy_action = self.scheduler.add_noise(action, noise, t)

        pred_noise = self.noise_pred(noisy_action, t, obs_cond)
        return nn.functional.mse_loss(pred_noise, noise)

    # ── inference ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_action(
        self,
        obs: torch.Tensor,          # (B, obs_horizon, *obs_shape)
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Denoise a random action sequence conditioned on obs.

        Args:
            obs: Observation history batch.
            num_inference_steps: DDIM-style step count (fewer = faster).

        Returns:
            action: (B, action_horizon, action_dim) – the first action_horizon
                    steps of the denoised pred_horizon sequence.
        """
        B = obs.shape[0]
        obs = obs.to(self.device)
        obs_cond = self.obs_encoder(obs)

        self.scheduler.set_timesteps(num_inference_steps)

        # Start from pure noise
        action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device
        )

        for t in self.scheduler.timesteps:
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            pred_noise = self.noise_pred(action, t_batch, obs_cond)
            action = self.scheduler.step(pred_noise, t, action).prev_sample

        # Return only the steps we actually execute
        return action[:, : self.action_horizon]

    # ── checkpoint helpers ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save model weights to specified path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[policy] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        """Load model weights from a checkpoint file."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
        print(f"[policy] Loaded checkpoint ← {path}")


# ── convenience functions ──────────────────────────────────────────────────
def save_policy(policy: DiffusionPolicy, path: str) -> None:
    policy.save(path)


def load_policy(path: str, **policy_kwargs) -> DiffusionPolicy:
    """Instantiate a DiffusionPolicy and load weights from a checkpoint.

    All constructor kwargs must be supplied (they are not stored in the
    checkpoint to keep files small).

    Example
    -------
    >>> from utils.project import get_checkpoint_dir
    >>> import os
    >>> policy = load_policy(
    ...     os.path.join(get_checkpoint_dir(), "epoch_50.pt"),
    ...     obs_mode="image", obs_horizon=2, obs_shape=(96, 96, 3),
    ...     action_dim=2, pred_horizon=16, action_horizon=8,
    ... )
    """
    policy = DiffusionPolicy(**policy_kwargs)
    policy.load(path)
    return policy
