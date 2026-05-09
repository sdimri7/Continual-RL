"""
Diffusion Policy training loop.

Designed for Google Colab:
* Checkpoints are saved to Drive on every `save_every` epoch.
* Resumes seamlessly from the latest checkpoint if `resume_from` is set.
* TensorBoard logs go to a Drive-backed directory so you can open them
  from a fresh session.
* Evaluation generates video files on Drive so you can watch them
  without re-running.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data.dataset import DiffusionPolicyDataset, Normalizer, PushTDemoDataset
from envs.make_env import make_eval_env
from policy.diffusion_policy import DiffusionPolicy
from training.config import Config


class Trainer:
    """Manages the full train / eval / checkpoint lifecycle.

    Args:
        config: Combined Config object (env + policy + train settings).
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[trainer] Using device: {self.device}")

        self._build_dataset()
        self._build_policy()
        self._build_optimizer()
        self._build_writer()

        self.start_epoch = 0
        if self.cfg.train.resume_from:
            self._resume(self.cfg.train.resume_from)

    # ── setup ──────────────────────────────────────────────────────────────
    def _build_dataset(self) -> None:
        tc, ec, pc = self.cfg.train, self.cfg.env, self.cfg.policy

        base = PushTDemoDataset(
            traj_path=tc.demo_path,
            obs_horizon=pc.obs_horizon,
            pred_horizon=pc.pred_horizon,
            obs_key=(
                "obs/sensor_data/base_camera/rgb"
                if ec.obs_mode == "rgb"
                else "obs/agent/qpos"
            ),
            image_size=(ec.image_size, ec.image_size) if ec.obs_mode == "rgb" else None,
        )

        self.normalizer = Normalizer(save_path=tc.normalizer_path)
        if os.path.exists(tc.normalizer_path):
            self.normalizer.load()
        else:
            self.normalizer.fit(base)

        dataset = DiffusionPolicyDataset(base, self.normalizer)
        self.loader = DataLoader(
            dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        print(f"[trainer] Dataset ready: {len(dataset)} samples.")

    def _build_policy(self) -> None:
        ec, pc = self.cfg.env, self.cfg.policy
        obs_shape = (
            (ec.image_size, ec.image_size, 3) if ec.obs_mode == "rgb" else (None,)
        )

        # Determine state dim from dataset if obs_mode is "state"
        if ec.obs_mode == "state":
            sample = self.loader.dataset[0]
            state_dim = sample["obs"].shape[-1]
            obs_shape = (state_dim,)

        # Infer action_dim
        action_dim = self.loader.dataset[0]["action"].shape[-1]

        self.policy = DiffusionPolicy(
            obs_mode=ec.obs_mode,
            obs_horizon=pc.obs_horizon,
            obs_shape=obs_shape,
            action_dim=action_dim,
            pred_horizon=pc.pred_horizon,
            action_horizon=pc.action_horizon,
            obs_cond_dim=pc.obs_cond_dim,
            num_diffusion_steps=pc.num_diffusion_steps,
            device=self.device,
        )
        total = sum(p.numel() for p in self.policy.parameters()) / 1e6
        print(f"[trainer] Policy ready: {total:.2f}M parameters.")

    def _build_optimizer(self) -> None:
        tc = self.cfg.train
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        # Use warmup + cosine annealing scheduler as per the Diffusion Policy paper
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,  # Start at 1% of lr
            end_factor=1.0,
            total_iters=tc.lr_warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=tc.num_epochs
        )
        # Chain: warmup first, then cosine annealing
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[tc.lr_warmup_steps],
        )

    def _build_writer(self) -> None:
        if self.cfg.train.use_tensorboard:
            os.makedirs(self.cfg.train.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.cfg.train.log_dir)
        else:
            self.writer = None

    # ── checkpoint ─────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int) -> str:
        os.makedirs(self.cfg.train.ckpt_dir, exist_ok=True)
        path = os.path.join(self.cfg.train.ckpt_dir, f"epoch_{epoch:04d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )
        print(f"[trainer] Checkpoint saved → {path}")
        return path

    def _resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        print(f"[trainer] Resumed from epoch {ckpt['epoch']} ← {path}")

    # ── evaluation ─────────────────────────────────────────────────────────
    def evaluate(self, epoch: int, record_video: bool = True) -> dict[str, float]:
        """Roll out the current policy and report success rate + mean reward + eval loss."""
        ec, pc, tc = self.cfg.env, self.cfg.policy, self.cfg.train

        video_dir = os.path.join(tc.eval_video_dir, f"epoch_{epoch:04d}") if record_video else None
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        env = make_eval_env(
            num_envs=ec.eval_num_envs,
            obs_mode=ec.obs_mode,
            record_dir=video_dir,
        )

        self.policy.eval()
        obs_buffer: list[torch.Tensor] = []
        obs, _ = env.reset(seed=epoch)

        total_reward = 0.0
        successes = 0
        num_episodes = 0
        steps = 0
        max_steps = 200
        episode_success = False

        while steps < max_steps:
            img = self._extract_obs(obs)
            obs_buffer.append(img)
            if len(obs_buffer) > pc.obs_horizon:
                obs_buffer.pop(0)

            # Pad buffer at start of episode
            while len(obs_buffer) < pc.obs_horizon:
                obs_buffer.insert(0, obs_buffer[0])

            obs_seq = torch.stack(obs_buffer, dim=1)  # (B, T, …)
            actions = self.policy.predict_action(obs_seq)  # (B, action_horizon, action_dim)

            for a_idx in range(actions.shape[1]):
                action = actions[:, a_idx]
                action_np = self.normalizer.unnormalize_action(action).cpu().numpy()
                obs, rew, term, trunc, info = env.step(
                    torch.from_numpy(action_np).to(self.device)
                )
                total_reward += rew.mean().item()
                if "success" in info and info["success"].any():
                    episode_success = True
                steps += 1
                if (term | trunc).any():
                    num_episodes += 1
                    successes += int(episode_success)
                    episode_success = False
                    obs_buffer = []  # clear stale frames before next episode
                    obs, _ = env.reset()
                    break

        # Count any episode still in progress at the step cap as one episode
        if num_episodes == 0:
            num_episodes = 1

        env.close()
        self.policy.train()

        # Compute eval loss on a sample of the training data
        eval_loss = self._compute_eval_loss(num_batches=10)

        metrics = {
            "eval/mean_reward": total_reward / steps,
            "eval/success_rate": successes / num_episodes,
            "eval/loss": eval_loss,
        }
        print(f"[trainer] Eval epoch {epoch}: {metrics}")
        return metrics

    def _compute_eval_loss(self, num_batches: int = 10) -> float:
        """Compute average loss on a sample of training data for evaluation."""
        from torch.utils.data import DataLoader
        
        eval_loader = DataLoader(
            self.loader.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        
        total_loss = 0.0
        count = 0
        
        self.policy.eval()
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= num_batches:
                    break
                obs = batch["obs"].to(self.device)
                action = batch["action"].to(self.device)
                loss = self.policy.compute_loss(obs, action)
                total_loss += loss.item()
                count += 1
        
        self.policy.train()
        return total_loss / count if count > 0 else 0.0

    # ── training loop ──────────────────────────────────────────────────────
    def train(self) -> None:
        """Run the full training loop from start_epoch to num_epochs."""
        tc = self.cfg.train
        self.policy.train()

        for epoch in range(self.start_epoch, tc.num_epochs):
            epoch_loss = 0.0

            pbar = tqdm(self.loader, desc=f"Epoch {epoch}/{tc.num_epochs}", leave=False)
            for batch in pbar:
                obs = batch["obs"].to(self.device)
                action = batch["action"].to(self.device)

                self.optimizer.zero_grad()
                loss = self.policy.compute_loss(obs, action)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()
            avg_loss = epoch_loss / len(self.loader)
            print(f"[trainer] Epoch {epoch:04d}  loss={avg_loss:.4f}  lr={self.scheduler.get_last_lr()[0]:.2e}")

            if self.writer:
                self.writer.add_scalar("train/loss", avg_loss, epoch)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch)

            if (epoch + 1) % tc.save_every == 0 or epoch == tc.num_epochs - 1:
                self._save_checkpoint(epoch)

            if (epoch + 1) % tc.eval_every == 0:
                record_video = (epoch + 1) % tc.eval_video_every == 0
                metrics = self.evaluate(epoch, record_video=record_video)
                if self.writer:
                    for k, v in metrics.items():
                        self.writer.add_scalar(k, v, epoch)

        if self.writer:
            self.writer.close()
        print("[trainer] Training complete.")

    # ── obs extraction helper ──────────────────────────────────────────────
    def _extract_obs(self, obs) -> torch.Tensor:
        # Handle both RGB (dict) and state (tensor) observation modes
        if isinstance(obs, dict):
            if "sensor_data" in obs:
                img = obs["sensor_data"]["base_camera"]["rgb"]  # (B, H, W, C) uint8
            elif "rgb" in obs:
                img = obs["rgb"]
            else:
                # Assume obs contains the tensor directly
                img = obs
        else:
            # State mode - obs is already a tensor
            img = obs
        
        # Normalize RGB images to [0, 1]
        if img.dtype == torch.uint8 or img.max() > 1.5:
            img = img.float() / 255.0
        

        # Resize images to match training resolution (96x96 by default)
        if self.cfg.env.obs_mode == "rgb" and img.ndim == 4:
            B, H, W, C = img.shape
            target_size = (self.cfg.env.image_size, self.cfg.env.image_size)
            if H != target_size[0] or W != target_size[1]:
                # Reshape to (B, C, H, W) for interpolate, resize, reshape back
                img = img.permute(0, 3, 1, 2)  # (B, C, H, W)
                img = torch.nn.functional.interpolate(
                    img, size=target_size, mode="bilinear", align_corners=False
                )
                img = img.permute(0, 2, 3, 1)  # (B, H, W, C)
        return img.to(self.device)
