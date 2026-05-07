# -*- coding: utf-8 -*-
"""
runner.py - CLI entry point for Diffusion Policy on Push-T.

Mirrors every stage in run.ipynb so you can run the full pipeline
(or individual stages) directly from the terminal:

    python runner.py setup
    python runner.py sanity
    python runner.py download
    python runner.py convert
    python runner.py train
    python runner.py train --resume-from checkpoints/epoch_0050.pt
    python runner.py eval
    python runner.py replay --episode 0

Run `python runner.py --help` or `python runner.py <stage> --help`
for all available flags.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


# ── project root on sys.path ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default data root — override with --project-dir
DEFAULT_PROJECT_DIR = os.path.join(os.path.expanduser("~"), "continual_rl")


# ══════════════════════════════════════════════════════════════════════════════
# Stages
# ══════════════════════════════════════════════════════════════════════════════

def stage_setup(args: argparse.Namespace) -> None:
    """Install dependencies (Vulkan on Colab, pip packages everywhere)."""
    from setup import full_setup
    full_setup()


def stage_sanity(args: argparse.Namespace) -> None:
    """Create a Push-T env, step through 10 random actions, save a frame grid."""
    import torch
    import matplotlib
    matplotlib.use("Agg")  # headless — no display needed
    import matplotlib.pyplot as plt

    from envs import make_pusht_env

    print(f"[sanity] Creating Push-T env with {args.num_envs} parallel envs …")
    env = make_pusht_env(num_envs=args.num_envs, obs_mode="rgb")
    obs, _ = env.reset(seed=0)
    env.unwrapped.print_sim_details()

    for _ in range(10):
        action = torch.from_numpy(env.action_space.sample())
        obs, rew, term, trunc, info = env.step(action)

    rgbs = obs["sensor_data"]["base_camera"]["rgb"]  # (B, H, W, C)
    n = min(args.num_envs, 4)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axs if n > 1 else [axs]):
        ax.imshow(rgbs[i].cpu().numpy())
        ax.axis("off")
    plt.suptitle("Push-T sanity check")
    out = os.path.join(args.project_dir, "sanity_check.png")
    os.makedirs(args.project_dir, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    env.close()
    print(f"[sanity] Frame grid saved to {out}")
    print("[sanity] Sanity check passed ✓")


def stage_download(args: argparse.Namespace) -> None:
    """Download Push-T expert demonstrations."""
    from data import download_demos

    demo_dir = download_demos(
        env_id=args.env_id,
        output_dir=os.path.join(args.project_dir, "demos"),
        force=args.force,
    )
    print(f"[download] Demos at: {demo_dir}")


def stage_convert(args: argparse.Namespace) -> None:
    """Convert raw demos to rgb + pd_ee_delta_pos format."""
    from data.demo_loader import convert_demos

    raw = os.path.join(
        args.project_dir, "demos", args.env_id, "motionplanning", "trajectory.h5"
    )
    converted = raw.replace("trajectory.h5", "trajectory.rgb.pd_ee_delta_pos.cpu.h5")

    if not args.force and os.path.exists(converted):
        print(f"[convert] Converted file already exists: {converted}")
        return

    convert_demos(
        traj_path=raw,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pos",
        num_procs=args.num_procs,
    )


def stage_train(args: argparse.Namespace) -> None:
    """Train the Diffusion Policy."""
    from training.config import Config, EnvConfig, PolicyConfig, TrainConfig
    from training.trainer import Trainer

    project_dir = args.project_dir

    # Load existing config from disk if available and not overriding
    config_path = os.path.join(project_dir, "config.json")
    if not args.new_config and os.path.exists(config_path):
        print(f"[train] Loading config from {config_path}")
        cfg = Config.load(config_path)
    else:
        converted_traj = os.path.join(
            project_dir,
            "demos", args.env_id, "motionplanning",
            "trajectory.rgb.pd_ee_delta_pos.cpu.h5",
        )
        cfg = Config(
            env=EnvConfig(
                env_id=args.env_id,
                num_envs=args.num_envs,
                eval_num_envs=args.eval_num_envs,
                obs_mode=args.obs_mode,
                image_size=args.image_size,
            ),
            policy=PolicyConfig(
                obs_horizon=args.obs_horizon,
                pred_horizon=args.pred_horizon,
                action_horizon=args.action_horizon,
                obs_cond_dim=args.obs_cond_dim,
                num_diffusion_steps=args.num_diffusion_steps,
            ),
            train=TrainConfig(
                demo_path=converted_traj,
                normalizer_path=os.path.join(project_dir, "normalizer_stats.npz"),
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                ckpt_dir=os.path.join(project_dir, "checkpoints"),
                save_every=args.save_every,
                resume_from=args.resume_from,
                eval_every=args.eval_every,
                eval_video_dir=os.path.join(project_dir, "eval_videos"),
                log_dir=os.path.join(project_dir, "runs"),
            ),
        )
        cfg.save(config_path)

    if args.resume_from:
        cfg.train.resume_from = args.resume_from

    print(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def stage_eval(args: argparse.Namespace) -> None:
    """Evaluate the latest (or specified) checkpoint."""
    from training.config import Config
    from training.trainer import Trainer

    project_dir = args.project_dir
    config_path = os.path.join(project_dir, "config.json")

    if not os.path.exists(config_path):
        sys.exit(f"[eval] No config.json found at {config_path}. Run 'train' first.")

    cfg = Config.load(config_path)

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpts = sorted(glob.glob(os.path.join(cfg.train.ckpt_dir, "epoch_*.pt")))
        if not ckpts:
            sys.exit("[eval] No checkpoints found. Run 'train' first.")
        ckpt = ckpts[-1]

    print(f"[eval] Using checkpoint: {ckpt}")
    cfg.train.resume_from = ckpt
    trainer = Trainer(cfg)
    epoch = int(os.path.basename(ckpt).split("epoch_")[1].split(".")[0])
    metrics = trainer.evaluate(epoch=epoch)
    print("[eval] Results:", metrics)


def stage_replay(args: argparse.Namespace) -> None:
    """Replay a single expert demo and save as a video."""
    from data import load_demo_metadata, replay_episode

    traj_path = os.path.join(
        args.project_dir, "demos", args.env_id, "motionplanning", "trajectory.h5"
    )
    if not os.path.exists(traj_path):
        sys.exit(f"[replay] Demo file not found: {traj_path}\nRun 'download' first.")

    h5, meta = load_demo_metadata(traj_path)
    n_eps = len(meta["episodes"])
    print(f"[replay] {n_eps} episodes available. Replaying episode {args.episode} …")

    video_path = replay_episode(
        episode_idx=args.episode,
        h5_file=h5,
        json_data=meta,
        save_dir=os.path.join(args.project_dir, "replays"),
    )
    print(f"[replay] Saved to: {video_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Diffusion Policy on Push-T — CLI runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project-dir",
        default=DEFAULT_PROJECT_DIR,
        help="Root directory for all outputs (demos, checkpoints, logs, videos).",
    )

    sub = parser.add_subparsers(dest="stage", required=True)

    # ── setup ──────────────────────────────────────────────────────────────
    sub.add_parser("setup", help="Install dependencies.")

    # ── sanity ─────────────────────────────────────────────────────────────
    p_sanity = sub.add_parser("sanity", help="Quick env smoke-test.")
    p_sanity.add_argument("--num-envs", type=int, default=4)

    # ── download ───────────────────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download expert demos.")
    p_dl.add_argument("--env-id", default="PushT-v1")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if present.")

    # ── convert ────────────────────────────────────────────────────────────
    p_cv = sub.add_parser("convert", help="Convert demos to rgb + pd_ee_delta_pos.")
    p_cv.add_argument("--env-id", default="PushT-v1")
    p_cv.add_argument("--num-procs", type=int, default=2)
    p_cv.add_argument("--force", action="store_true")

    # ── train ──────────────────────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="Train the Diffusion Policy.")
    p_tr.add_argument("--env-id", default="PushT-v1")
    p_tr.add_argument("--obs-mode", default="rgb", choices=["rgb", "state"])
    p_tr.add_argument("--num-envs", type=int, default=64)
    p_tr.add_argument("--eval-num-envs", type=int, default=10)
    p_tr.add_argument("--image-size", type=int, default=96)
    p_tr.add_argument("--obs-horizon", type=int, default=2)
    p_tr.add_argument("--pred-horizon", type=int, default=16)
    p_tr.add_argument("--action-horizon", type=int, default=8)
    p_tr.add_argument("--obs-cond-dim", type=int, default=256)
    p_tr.add_argument("--num-diffusion-steps", type=int, default=100)
    p_tr.add_argument("--num-epochs", type=int, default=100)
    p_tr.add_argument("--batch-size", type=int, default=256)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--save-every", type=int, default=10)
    p_tr.add_argument("--eval-every", type=int, default=10)
    p_tr.add_argument(
        "--resume-from",
        default=None,
        metavar="PATH",
        help="Path to a .pt checkpoint to resume training from.",
    )
    p_tr.add_argument(
        "--new-config",
        action="store_true",
        help="Ignore existing config.json and build a fresh one from CLI flags.",
    )

    # ── eval ───────────────────────────────────────────────────────────────
    p_ev = sub.add_parser("eval", help="Evaluate the latest checkpoint.")
    p_ev.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to a specific .pt checkpoint (default: latest in checkpoints/).",
    )

    # ── replay ─────────────────────────────────────────────────────────────
    p_rp = sub.add_parser("replay", help="Replay an expert demo and save video.")
    p_rp.add_argument("--env-id", default="PushT-v1")
    p_rp.add_argument("--episode", type=int, default=0, help="Episode index to replay.")

    return parser


# ── dispatch ───────────────────────────────────────────────────────────────
_STAGES = {
    "setup":    stage_setup,
    "sanity":   stage_sanity,
    "download": stage_download,
    "convert":  stage_convert,
    "train":    stage_train,
    "eval":     stage_eval,
    "replay":   stage_replay,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Normalise hyphenated attr names to underscored (argparse quirk)
    args.project_dir = args.project_dir
    if hasattr(args, "env_id") and not hasattr(args, "env-id"):
        pass  # already underscore form

    fn = _STAGES[args.stage]
    fn(args)


if __name__ == "__main__":
    main()
