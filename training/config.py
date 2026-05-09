import datetime
import os
from dataclasses import asdict, dataclass, field


@dataclass
class EnvConfig:
    """Environment configuration."""
    env_id: str = "PushT-v1"
    num_envs: int = 64
    eval_num_envs: int = 10
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pos"
    reward_mode: str = "dense"
    image_size: int = 96


@dataclass
class PolicyConfig:
    """Policy configuration."""
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8
    obs_cond_dim: int = 256
    num_diffusion_steps: int = 100


@dataclass
class TrainConfig:
    """Training configuration.
    
    Paths are relative to project root and should be set using utils/project.py
    helper functions when creating this config (typically via runner.py).
    """
    
    # data - these should be set programmatically, not hardcoded
    demo_path: str = ""
    normalizer_path: str = ""

    # optimisation
    num_epochs: int = 3000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 500

    # checkpointing (Base directories)
    ckpt_dir: str = ""
    save_every: int = 200
    resume_from: str | None = None

    # evaluation
    eval_every: int = 50
    eval_episodes: int = 10
    eval_video_every: int = 50 # New parameter for video saving frequency
    eval_video_dir: str = ""

    # logging
    log_dir: str = ""
    use_tensorboard: bool = True
    
    # Run identifier for unique subdirectories
    run_id: str = ""
    
    @property
    def is_configured(self) -> bool:
        """Check if all required paths are set."""
        return bool(self.demo_path and self.ckpt_dir and self.log_dir)

    def __post_init__(self):
        """Automatically creates unique subdirectories after initialization."""
        # Only modify log/ckpt/eval_video dirs if they have base values set
        if not self.run_id:
            opt_name = f"lr{self.lr:.1e}_bs{self.batch_size}_wd{self.weight_decay:.1e}"
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            self.run_id = f"{opt_name}_{timestamp}"
        
        # Update directories to include the unique run name
        if self.log_dir:
            self.log_dir = os.path.join(self.log_dir, self.run_id)
        if self.ckpt_dir:
            self.ckpt_dir = os.path.join(self.ckpt_dir, self.run_id)
        if self.eval_video_dir:
            self.eval_video_dir = os.path.join(self.eval_video_dir, self.run_id)


@dataclass
class Config:
    """Combined configuration for training."""
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "env": asdict(self.env),
            "policy": asdict(self.policy),
            "train": asdict(self.train),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            env=EnvConfig(**d.get("env", {})),
            policy=PolicyConfig(**d.get("policy", {})),
            train=TrainConfig(**d.get("train", {})),
        )
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load config from JSON file.
        
        When loading from a saved config, the paths are already complete (including run_id
        subdirectory), so we skip the __post_init__ path modification to avoid double-appending.
        """
        import json
        with open(path, "r") as f:
            d = json.load(f)
        
        train_data = d.get("train", {})
        
        # Extract the complete paths from the loaded config (they include run_id subdirectory)
        complete_ckpt_dir = train_data.get("ckpt_dir", "")
        complete_eval_dir = train_data.get("eval_video_dir", "")
        complete_log_dir = train_data.get("log_dir", "")
        
        # Create config with empty base directories so __post_init__ won't modify them
        train_config = TrainConfig(
            demo_path=train_data.get("demo_path", ""),
            normalizer_path=train_data.get("normalizer_path", ""),
            num_epochs=train_data.get("num_epochs", 3000),
            batch_size=train_data.get("batch_size", 256),
            lr=train_data.get("lr", 1e-4),
            weight_decay=train_data.get("weight_decay", 1e-6),
            lr_warmup_steps=train_data.get("lr_warmup_steps", 500),
            ckpt_dir="",  # Empty so __post_init__ won't modify
            save_every=train_data.get("save_every", 200),
            resume_from=train_data.get("resume_from"),
            eval_every=train_data.get("eval_every", 50),
            eval_episodes=train_data.get("eval_episodes", 10),
            eval_video_every=train_data.get("eval_video_every", 50), # New parameter for video saving frequency
            eval_video_dir="",  # Empty so __post_init__ won't modify
            log_dir="",  # Empty so __post_init__ won't modify
            use_tensorboard=train_data.get("use_tensorboard", True),
            run_id=train_data.get("run_id", ""),
        )
        
        # Manually set the complete paths after __post_init__ has run
        train_config.ckpt_dir = complete_ckpt_dir
        train_config.eval_video_dir = complete_eval_dir
        train_config.log_dir = complete_log_dir
        
        return cls(
            env=EnvConfig(**d.get("env", {})),
            policy=PolicyConfig(**d.get("policy", {})),
            train=train_config,
        )
    
    def __repr__(self) -> str:
        return f"Config(env={self.env}, policy={self.policy}, train={self.train})"
