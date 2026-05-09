"""
Diffusion Policy: wraps the UNet noise predictor with a DDPM scheduler.

Reference: Chi et al. 2023, https://arxiv.org/abs/2303.04137

Matches the official ManiSkill diffusion policy baseline (train_rgbd.py):
- Visual features are encoded per-frame and concatenated with proprioceptive
  state before flattening across obs_horizon.
- Action chunk slice starts at obs_horizon - 1 (not 0) so the first executed
  action corresponds to the current timestep.
- Actions are assumed to be in [-1, 1] (pd_ee_delta_pos guarantee); no
  external normaliser is needed.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .networks import ConditionalUNet1D, ObservationEncoder


class DiffusionPolicy(nn.Module):
    """End-to-end Diffusion Policy model.

    Components
    ----------
    * ``obs_encoder``  – encodes (image, state) history into a conditioning
                         vector: obs_horizon × (visual_feature_dim + state_dim).
    * ``noise_pred``   – conditional 1-D U-Net predicting ε given
                         (noisy_action, timestep, obs_cond).
    * ``scheduler``    – DDPM noise scheduler (training) / DDIM (inference).

    Args:
        obs_mode:           "rgb" / "image" for pixel input, "state" for flat vector.
        obs_horizon:        History length used as conditioning.
        obs_shape:          Shape of a single observation (H, W, C) or (state_dim,).
        state_dim:          Dimension of proprioceptive state concatenated per
                            timestep alongside the visual feature (0 = no state).
        action_dim:         Dimensionality of a single action vector.
        pred_horizon:       Total predicted action steps.
        action_horizon:     Steps actually executed from each prediction.
        obs_cond_dim:       Per-frame visual feature size (output of ResNet encoder).
        num_diffusion_steps: Total DDPM timesteps T.
        device:             "cuda" or "cpu".
    """

    def __init__(
        self,
        obs_mode: str = "rgb",
        obs_horizon: int = 2,
        obs_shape: tuple[int, ...] = (96, 96, 3),
        state_dim: int = 0,
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.device = device

        self.obs_encoder = ObservationEncoder(
            obs_mode=obs_mode,
            obs_horizon=obs_horizon,
            obs_shape=obs_shape,
            visual_feature_dim=obs_cond_dim,
            state_dim=state_dim,
        )

        # The UNet conditioning dim equals the encoder's full output dim
        unet_cond_dim = self.obs_encoder.out_dim

        self.noise_pred = ConditionalUNet1D(
            action_dim=action_dim,
            pred_horizon=pred_horizon,
            obs_cond_dim=unet_cond_dim,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self.to(device)

    # ── training step ──────────────────────────────────────────────────────
    def compute_loss(
        self,
        obs: torch.Tensor,                   # (B, obs_horizon, H, W, C)
        action: torch.Tensor,                # (B, pred_horizon, action_dim)
        state: torch.Tensor | None = None,   # (B, obs_horizon, state_dim)
    ) -> torch.Tensor:
        """DDPM training loss: predict ε from the noisy action at a random t."""
        B = obs.shape[0]
        obs = obs.to(self.device)
        action = action.to(self.device)
        if state is not None:
            state = state.to(self.device)

        obs_cond = self.obs_encoder(obs, state)

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
        obs: torch.Tensor,                   # (B, obs_horizon, H, W, C)
        state: torch.Tensor | None = None,   # (B, obs_horizon, state_dim)
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        """Denoise a random action sequence conditioned on obs and state.

        Action chunk slice follows the official baseline:
            start = obs_horizon - 1
            end   = start + action_horizon
        so the first executed action aligns with the current timestep.

        Returns:
            action: (B, action_horizon, action_dim) in [-1, 1]
        """
        B = obs.shape[0]
        obs = obs.to(self.device)
        if state is not None:
            state = state.to(self.device)
        obs_cond = self.obs_encoder(obs, state)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        action = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device
        )

        for t in self.inference_scheduler.timesteps:
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            pred_noise = self.noise_pred(action, t_batch, obs_cond)
            action = self.inference_scheduler.step(pred_noise, t, action).prev_sample

        # Return the action_horizon-length chunk aligned to the current timestep
        start = self.obs_horizon - 1
        return action[:, start : start + self.action_horizon]

    # ── checkpoint helpers ─────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[policy] Saved checkpoint → {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state)
        print(f"[policy] Loaded checkpoint ← {path}")


# ── convenience functions ──────────────────────────────────────────────────
def save_policy(policy: DiffusionPolicy, path: str) -> None:
    policy.save(path)


def load_policy(path: str, **policy_kwargs) -> DiffusionPolicy:
    """Instantiate a DiffusionPolicy and load weights from a checkpoint."""
    policy = DiffusionPolicy(**policy_kwargs)
    policy.load(path)
    return policy
