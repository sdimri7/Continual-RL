import datetime
import os
from dataclasses import asdict, dataclass, field

# ... (EnvConfig and PolicyConfig remain the same)

@dataclass
class TrainConfig:
    
    # data
    demo_path: str = "/content/drive/MyDrive/continual_rl/demos/PushT-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.cpu.h5"
    normalizer_path: str = "/content/drive/MyDrive/continual_rl/normalizer_stats.npz"

    # optimisation
    num_epochs: int = 3000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 500

    # checkpointing (Base directories)
    ckpt_dir: str = "/content/drive/MyDrive/continual_rl/checkpoints"
    save_every: int = 200
    resume_from: str | None = None

    # evaluation
    eval_every: int = 50
    eval_episodes: int = 10
    eval_video_dir: str = "/content/drive/MyDrive/continual_rl/eval_videos"

    # logging
    log_dir: str = "/content/drive/MyDrive/continual_rl/runs"
    use_tensorboard: bool = True

    def __post_init__(self):
        """Automatically creates unique subdirectories after initialization."""
        opt_name = f"lr{self.lr:.1e}_bs{self.batch_size}_wd{self.weight_decay:.1e}"
        
        # 2. Add a timestamp to avoid overwriting if you run the same params twice
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        
        run_id = f"{opt_name}_{timestamp}"
        
        # Update directories to include the unique run name
        self.log_dir = os.path.join(self.log_dir, self.run_id)
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.run_id)
        self.eval_video_dir = os.path.join(self.eval_video_dir, self.run_id)