"""
Dataclass-based configuration for all training components.

Keeping everything in one place makes it easy to serialise, version,
and reproduce runs.  Save configs with ``config.save()`` to Drive.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


# ── environment ────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    env_id: str = "PushT-v1"
    num_envs: int = 64            # parallel training envs
    eval_num_envs: int = 10       # parallel evaluation envs
    obs_mode: str = "rgb"         # "rgb" | "state"
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "dense"
    image_size: int = 96          # resize images to (image_size, image_size)


# ── policy / model ─────────────────────────────────────────────────────────
@dataclass
class PolicyConfig:
    obs_horizon: int = 2          # frames of history used as conditioning
    pred_horizon: int = 16        # steps of future actions predicted
    action_horizon: int = 8       # steps of predicted actions to execute
    obs_cond_dim: int = 256       # observation encoder output size
    num_diffusion_steps: int = 100


# ── training loop ──────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # data
    demo_path: str = "/content/drive/MyDrive/continual_rl/demos/PushT-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.cpu.h5"
    normalizer_path: str = "/content/drive/MyDrive/continual_rl/normalizer_stats.npz"

    # optimisation
    num_epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 500

    # checkpointing
    ckpt_dir: str = "/content/drive/MyDrive/continual_rl/checkpoints"
    save_every: int = 10          # save checkpoint every N epochs
    resume_from: str | None = None  # path to a .pt file to resume from

    # evaluation
    eval_every: int = 10          # run eval every N epochs
    eval_episodes: int = 10       # number of full episodes to evaluate
    eval_video_dir: str = "/content/drive/MyDrive/continual_rl/eval_videos"

    # logging
    log_dir: str = "/content/drive/MyDrive/continual_rl/runs"
    use_tensorboard: bool = True


# ── combined config ────────────────────────────────────────────────────────
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: str) -> None:
        """Serialise to JSON and write to path (creates parent dirs)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[config] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Config":
        """Deserialise from a JSON file saved by Config.save()."""
        with open(path) as f:
            data = json.load(f)
        cfg = cls(
            env=EnvConfig(**data["env"]),
            policy=PolicyConfig(**data["policy"]),
            train=TrainConfig(**data["train"]),
        )
        print(f"[config] Loaded from {path}")
        return cfg

    def __repr__(self) -> str:
        return json.dumps(asdict(self), indent=2)
