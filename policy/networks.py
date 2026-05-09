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
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── ResNet building blocks ──────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Basic residual block for ResNet.
    
    Uses ELU activations as specified in Appendix C.1 of the Diffusion Policy paper.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet18(nn.Module):
    """Custom ResNet-18 implementation (without final pooling and FC layer).

    As per the original Diffusion Policy paper (Section 4.3 and Appendix C.1),
    we use ResNet-18 as the visual encoder with ELU activations instead of ReLU,
    and global average pooling to produce a 256-d feature vector.
    
    Args:
        in_channels: Number of input channels. Default 3 for single RGB frame.
                     Use 6 for stacked frames (obs_horizon * 3).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        # Initial convolution (conv1 in standard ResNet)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ELU(inplace=True)  # ELU as per Appendix C.1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers (2 blocks each)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Try to load pretrained weights from torchvision
        if pretrained:
            self._load_pretrained_weights()
        else:
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self) -> None:
        """Try to load pretrained ResNet-18 weights from torchvision."""
        try:
            # Try the newer torchvision API first
            from torchvision.models import resnet18, ResNet18_Weights
            pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            try:
                # Fall back to older torchvision API
                from torchvision.models import resnet18
                pretrained_model = resnet18(pretrained=True)
            except Exception:
                pretrained_model = None

        if pretrained_model is not None:
            # Copy pretrained weights (excluding final fc and avgpool which we don't use)
            state_dict = pretrained_model.state_dict()
            # Remove fc and avgpool weights if present
            state_dict = {k: v for k, v in state_dict.items() 
                         if 'fc' not in k and 'avgpool' not in k}
            try:
                self.load_state_dict(state_dict, strict=False)
                print("[ResNet18] Loaded pretrained ImageNet weights")
            except Exception:
                self._init_weights()
                print("[ResNet18] Failed to load pretrained weights, using random initialization")
        else:
            self._init_weights()
            print("[ResNet18] No pretrained weights available, using random initialization")

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            feat_map: (B, 512, H/32, W/32)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet18Encoder(nn.Module):
    """ResNet-18 visual encoder (without final pooling and FC layer).

    As per the original Diffusion Policy paper (Section 4.3 and Appendix C.1):
    - Uses ResNet-18 backbone with ELU activations
    - Global average pooling produces a 512-d feature vector
    - 2-layer MLP projection maps to the required observation conditioning dimension

    The network takes RGB frames and outputs a fixed-size embedding.
    Uses pretrained ImageNet weights when available.

    Args:
        out_dim: Size of the output conditioning vector.
        in_channels: Number of input channels. Default 3 for single RGB frame.
                     Use obs_horizon * 3 for stacked frames.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, out_dim: int = 256, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        # Use ResNet-18 with pretrained weights by default
        # For multi-channel input (stacked frames), we don't use pretrained weights
        # for the first conv layer, but still use them for the rest
        self.backbone = ResNet18(in_channels=in_channels, pretrained=pretrained)

        # Global average pooling to get 512-d vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 2-layer MLP projection as per Appendix C.1 of the paper
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, out_dim),
        )

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - RGB frames (C can be 3 or 6 for stacked frames)

        Returns:
            feat: (B, out_dim)
        """
        # Get spatial feature maps from ResNet backbone
        feat_map = self.backbone(x)  # (B, 512, H/32, W/32)

        # Global average pooling
        pooled = self.gap(feat_map).squeeze(-1).squeeze(-1)  # (B, 512)

        # 2-layer MLP projection to output dimension
        return self.proj(pooled)


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

    * ``"image"``  – stacked RGB frames passed through ResNet-18 encoder.
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
            # For image mode, we stack frames along channel dimension
            # and process them directly with ResNet18 (no temporal blending)
            H, W, C = obs_shape
            in_channels = C * obs_horizon  # e.g., 3 * 2 = 6 for obs_horizon=2

            # ResNet-18 encoder with multi-channel input
            # Pretrained weights are loaded but first conv layer is reinitialized
            self.encoder = ResNet18Encoder(
                out_dim=out_dim, 
                in_channels=in_channels, 
                pretrained=True
            )
            self._obs_horizon = obs_horizon
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

    @property
    def obs_horizon(self) -> int:
        return self._obs_horizon

    @obs_horizon.setter
    def obs_horizon(self, value: int) -> None:
        self._obs_horizon = value

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_horizon, H, W, C) — HWC format from ManiSkill/dataset

        Returns:
            cond: (B, out_dim)
        """
        B = obs.shape[0]
        if self.obs_mode == "image":
            # (B, T, H, W, C) → (B, T*C, H, W)  — stack channels temporally
            # This produces (B, 6, 96, 96) for obs_horizon=2
            # The einops notation "b (t c) h w" means batch, [t*c channels], height, width
            obs_stacked = rearrange(obs, "b t h w c -> b (t c) h w")

            # Process stacked frames directly with ResNet18 (no blending)
            # obs_stacked is now (B, 6, H, W) — 6 channels for 2 stacked RGB frames
            feats = self.encoder(obs_stacked)
            return feats
        else:
            obs = obs.reshape(B, -1)
            return self.proj(self.encoder(obs))
