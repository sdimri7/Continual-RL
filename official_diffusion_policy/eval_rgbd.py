"""Standalone evaluation script for RGB-D Diffusion Policy checkpoints.

Evaluates a saved checkpoint independently from training, with support for
all the same environment/policy variables as train_rgbd.py.

Usage:
    # Basic evaluation
    python eval_rgbd.py \\
        --checkpoint runs/<exp_name>/checkpoints/best_eval_success_once.pt \\
        --env-id PushT-v1 \\
        --control-mode pd_ee_delta_pose \\
        --obs-mode rgb \\
        --max_episode_steps 150 \\
        --num-eval-envs 50 \\
        --num-eval-episodes 250

    # With video recording
    python eval_rgbd.py \\
        --checkpoint runs/<exp_name>/checkpoints/best_eval_success_once.pt \\
        --env-id PushT-v1 \\
        --control-mode pd_ee_delta_pose \\
        --obs-mode rgb \\
        --max_episode_steps 150 \\
        --num-eval-envs 50 \\
        --num-eval-episodes 250 \\
        --capture-video \\
        --video-dir eval_outputs/videos

    # GPU-parallel evaluation
    python eval_rgbd.py \\
        --checkpoint runs/<exp_name>/checkpoints/best_eval_success_once.pt \\
        --env-id PushT-v1 \\
        --control-mode pd_ee_delta_pose \\
        --obs-mode rgb \\
        --max_episode_steps 150 \\
        --sim-backend physx_cuda \\
        --num-eval-envs 100 \\
        --num-eval-episodes 500
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from gymnasium.vector.vector_env import VectorEnv
from gymnasium import spaces


@dataclass
class Args:
    # --- Required ---
    checkpoint: str = ""
    """Path to the .pt checkpoint file to evaluate"""

    # --- Environment ---
    env_id: str = "PushT-v1"
    """the id of the environment"""
    obs_mode: str = "rgb"
    """The observation mode to use. Can be 'rgb', 'depth', or 'rgb+depth'."""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode. Must match the control mode used during training."""
    max_episode_steps: Optional[int] = None
    """max episode steps for the environment. Must match training value."""
    sim_backend: str = "physx_cpu"
    """simulation backend: 'physx_cpu' or 'physx_cuda'"""

    # --- Evaluation ---
    num_eval_episodes: int = 100
    """total number of episodes to evaluate"""
    num_eval_envs: int = 10
    """number of parallel environments for evaluation"""
    seed: int = 1
    """random seed"""

    # --- Diffusion Policy architecture (must match training) ---
    obs_horizon: int = 2
    """observation horizon (history length)"""
    act_horizon: int = 8
    """action execution horizon"""
    pred_horizon: int = 16
    """action prediction horizon"""
    diffusion_step_embed_dim: int = 64
    """diffusion step embedding dimension"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """U-Net channel dimensions"""
    n_groups: int = 8
    """number of groups for GroupNorm in U-Net"""

    # --- Video / output ---
    capture_video: bool = False
    """whether to record evaluation videos"""
    video_dir: Optional[str] = None
    """directory to save videos (default: eval_outputs/<checkpoint_name>/videos)"""
    output_dir: Optional[str] = None
    """directory to save evaluation results CSV (default: eval_outputs/<checkpoint_name>)"""

    # --- Misc ---
    cuda: bool = True
    """use GPU if available"""
    torch_deterministic: bool = True
    """enable deterministic mode for reproducibility"""
    use_ema: bool = True
    """use EMA weights if available in checkpoint (recommended)"""


class Agent(torch.nn.Module):
    """RGB-D Diffusion Policy agent — identical architecture to train_rgbd.py."""

    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space["state"].shape) == 2
        assert len(env.single_action_space.shape) == 1
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels,
            out_dim=visual_feature_dim,
            pool_feature_map=True,
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)
        visual_feature = self.visual_encoder(img_seq)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature.flatten(start_dim=1)

    def get_action(self, obs_seq):
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(obs_seq, eval_mode=True)

            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]


if __name__ == "__main__":
    args = tyro.cli(Args)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required. Provide the path to a .pt file.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.max_episode_steps is None:
        raise ValueError(
            "--max_episode_steps is required (must match the value used during training)."
        )

    # Derive output directories from checkpoint path if not specified
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ckpt_parent = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
    default_output_base = f"eval_outputs/{ckpt_parent}_{ckpt_name}"

    if args.output_dir is None:
        args.output_dir = default_output_base
    if args.video_dir is None:
        args.video_dir = os.path.join(default_output_base, "videos")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Diffusion Policy Evaluation (RGB-D)")
    print("=" * 60)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Environment:       {args.env_id}")
    print(f"Obs mode:          {args.obs_mode}")
    print(f"Control mode:      {args.control_mode}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Sim backend:       {args.sim_backend}")
    print(f"Num eval envs:     {args.num_eval_envs}")
    print(f"Num eval episodes: {args.num_eval_episodes}")
    print(f"Use EMA weights:   {args.use_ema}")
    print(f"Capture video:     {args.capture_video}")
    print(f"Output dir:        {args.output_dir}")
    print(f"Seed:              {args.seed}")
    print("=" * 60)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # Create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    video_dir = args.video_dir if args.capture_video else None
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # Build agent with matching architecture
    agent = Agent(envs, args).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.use_ema and "ema_agent" in ckpt:
        agent.load_state_dict(ckpt["ema_agent"])
        print("Loaded EMA agent weights.")
    elif "agent" in ckpt:
        agent.load_state_dict(ckpt["agent"])
        print("Loaded agent weights.")
    else:
        # Try loading as a raw state_dict
        agent.load_state_dict(ckpt)
        print("Loaded raw state_dict weights.")

    param_count = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {param_count:,}")

    # Run evaluation
    print(f"\nRunning {args.num_eval_episodes} evaluation episodes...")
    start_time = time.time()

    eval_metrics = evaluate(
        args.num_eval_episodes, agent, envs, device, args.sim_backend
    )

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.1f}s")

    # Compute and display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    results = {}
    for k in eval_metrics.keys():
        values = eval_metrics[k]
        if isinstance(values, list):
            values = np.concatenate(values) if len(values) > 0 else np.array([])
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[k] = {"mean": float(mean_val), "std": float(std_val), "n": len(values)}
        print(f"  {k:<25} {mean_val:.4f} +/- {std_val:.4f}  (n={len(values)})")

    # Save results to CSV
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    with open(csv_path, "w") as f:
        f.write("metric,mean,std,n\n")
        for k, v in results.items():
            f.write(f"{k},{v['mean']:.6f},{v['std']:.6f},{v['n']}\n")
    print(f"\nResults saved to: {csv_path}")

    # Save full config for reproducibility
    config_path = os.path.join(args.output_dir, "eval_config.txt")
    with open(config_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"Config saved to:  {config_path}")

    if args.capture_video:
        import glob
        videos = glob.glob(os.path.join(args.video_dir, "**/*.mp4"), recursive=True)
        print(f"Videos saved:     {len(videos)} files in {args.video_dir}")

    envs.close()
    print("\nDone.")
