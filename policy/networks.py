"""
Neural network building blocks for Diffusion Policy.

Architecture follows Chi et al. 2023:
    "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    https://arxiv.org/abs/2303.04137

ConditionalUNet1D    – 1-D temporal U-Net with FiLM conditioning on
                        (obs_embedding + diffusion timestep).
ObservationEncoder   – encodes stacked RGB frames (or state vectors)
                        into a fixed-size embedding.
SinusoidalPosEmb     – sinusoidal diffusion-timestep embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── sinusoidal timestep embedding ──────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    """Maps an integer diffusion timestep to a continuous embedding vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── FiLM conditioning ──────────────────────────────────────────────────────
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale and shift a feature map.

    Receives a conditioning vector ``cond`` and produces per-channel
    scale (γ) and bias (β) applied to the input feature map ``x``.
    """

    def __init__(self, cond_dim: int, out_channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(cond_dim, out_channels)
        self.bias = nn.Linear(cond_dim, out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)   cond: (B, cond_dim)
        gamma = self.scale(cond).unsqueeze(-1)   # (B, C, 1)
        beta = self.bias(cond).unsqueeze(-1)     # (B, C, 1)
        return gamma * x + beta


# ── 1-D residual block ─────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """Single 1-D residual block with FiLM conditioning and GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
        self.film = FiLM(cond_dim, out_channels)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.film(h, cond)
        h = self.block2(h)
        return h + self.residual_conv(x)


# ── conditional 1-D U-Net ──────────────────────────────────────────────────
class ConditionalUNet1D(nn.Module):
    """1-D temporal U-Net that predicts the noise added to an action sequence.

    Conditioning is a concatenation of the obs embedding and the sinusoidal
    timestep embedding, applied via FiLM at every level of the U-Net.

    Args:
        action_dim: Dimensionality of a single action vector.
        pred_horizon: Number of steps in the action prediction window.
        obs_cond_dim: Size of the observation conditioning vector.
        channel_mults: Channel multipliers at each U-Net level.
        n_groups: GroupNorm group count.
    """

    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        obs_cond_dim: int,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        base_channels: int = 256,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        # timestep embedding
        t_dim = base_channels
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, t_dim),
        )
        cond_dim = t_dim + obs_cond_dim  # timestep + obs

        channels = [base_channels * m for m in channel_mults]

        # encoder (downsampling)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_ch = action_dim
        for ch in channels:
            self.enc_blocks.append(ResBlock1D(in_ch, ch, cond_dim, n_groups=n_groups))
            self.downs.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
            in_ch = ch

        # bottleneck
        self.mid = ResBlock1D(in_ch, in_ch, cond_dim, n_groups=n_groups)

        # decoder (upsampling)
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            self.ups.append(nn.ConvTranspose1d(in_ch, ch, 4, stride=2, padding=1))
            self.dec_blocks.append(ResBlock1D(ch * 2, ch, cond_dim, n_groups=n_groups))
            in_ch = ch

        self.out_conv = nn.Conv1d(in_ch, action_dim, 1)

    def forward(
        self,
        noisy_action: torch.Tensor,  # (B, pred_horizon, action_dim)
        timestep: torch.Tensor,       # (B,) int
        obs_cond: torch.Tensor,       # (B, obs_cond_dim)
    ) -> torch.Tensor:
        # (B, T, A) → (B, A, T) for conv
        x = rearrange(noisy_action, "b t a -> b a t")

        t_emb = self.time_emb(timestep)
        cond = torch.cat([t_emb, obs_cond], dim=-1)

        skips = []
        for enc, down in zip(self.enc_blocks, self.downs):
            x = enc(x, cond)
            skips.append(x)
            x = down(x)

        x = self.mid(x, cond)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = dec(x, cond)

        x = self.out_conv(x)
        return rearrange(x, "b a t -> b t a")


# ── observation encoder ────────────────────────────────────────────────────
class ObservationEncoder(nn.Module):
    """Encode a sequence of observations into a flat conditioning vector.

    Supports two modes:

    * ``"image"``  – stacked RGB frames passed through a lightweight CNN.
    * ``"state"``  – flat state vectors projected by an MLP.

    Args:
        obs_mode: "image" or "state".
        obs_horizon: Number of observations in the history window.
        obs_shape: Shape of a single observation.
            For images: (H, W, C).
            For state:  (state_dim,).
        out_dim: Size of the output conditioning vector.
    """

    def __init__(
        self,
        obs_mode: str,
        obs_horizon: int,
        obs_shape: tuple[int, ...],
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_mode = obs_mode
        self.obs_horizon = obs_horizon

        if obs_mode == "image":
            H, W, C = obs_shape
            in_channels = C * obs_horizon
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Compute CNN output size
            dummy = torch.zeros(1, in_channels, H, W)
            cnn_out = self.encoder(dummy).shape[-1]
            self.proj = nn.Linear(cnn_out, out_dim)
        else:
            state_dim = math.prod(obs_shape) * obs_horizon
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Mish(),
                nn.Linear(256, out_dim),
                nn.Mish(),
            )
            self.proj = nn.Identity()

        self.out_dim = out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_horizon, *obs_shape)

        Returns:
            cond: (B, out_dim)
        """
        B = obs.shape[0]
        if self.obs_mode == "image":
            # (B, T, H, W, C) → (B, T*C, H, W)
            obs = rearrange(obs, "b t h w c -> b (t c) h w")
            feats = self.encoder(obs)
            return self.proj(feats)
        else:
            obs = obs.reshape(B, -1)
            return self.proj(self.encoder(obs))
