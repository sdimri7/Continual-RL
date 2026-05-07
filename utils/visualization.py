"""
Visualization helpers for ManiSkill environments and Diffusion Policy.

All functions are designed to work inside Jupyter / Colab notebooks.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Video


# ── single-frame helpers ───────────────────────────────────────────────────
def show_rgb(tensor: torch.Tensor, title: str = "", env_id: int = 0) -> None:
    """Show a single RGB frame from a batched (B, H, W, C) tensor."""
    img = tensor[env_id].cpu().numpy()
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_camera_view(
    obs_camera: dict[str, torch.Tensor],
    title: str = "Camera",
    env_id: int = 0,
) -> None:
    """Show RGB, Depth, and Segmentation side-by-side for one parallel env.

    Args:
        obs_camera: Dict with keys "rgb", "depth", and optionally "segmentation".
        title: Plot title prefix.
        env_id: Which environment index to display.
    """
    rgb = obs_camera["rgb"][env_id].cpu().numpy()
    depth = obs_camera["depth"][..., 0][env_id].cpu().numpy()

    n_cols = 3 if "segmentation" in obs_camera else 2
    fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axs[0].imshow(rgb)
    axs[0].set_title(f"{title} – RGB")
    axs[0].axis("off")

    axs[1].imshow(depth, cmap="gray")
    axs[1].set_title(f"{title} – Depth")
    axs[1].axis("off")

    if "segmentation" in obs_camera:
        seg = obs_camera["segmentation"][..., 0][env_id].cpu().numpy()
        axs[2].imshow(seg)
        axs[2].set_title(f"{title} – Segmentation")
        axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_pointcloud(obs: dict[str, Any], env_id: int = 0) -> None:
    """Render a 3-D point cloud from a ManiSkill pointcloud observation.

    Requires `trimesh` (``pip install trimesh``).
    """
    try:
        import trimesh
    except ImportError:
        print("[viz] Install trimesh:  pip install trimesh")
        return

    v = obs["pointcloud"]["xyzw"][env_id, ..., :3].cpu().numpy()
    cam2world = obs["sensor_param"]["base_camera"]["cam2world_gl"][env_id].cpu().numpy()
    colors = obs["pointcloud"]["rgb"][env_id].cpu().numpy()

    camera = trimesh.scene.Camera(
        "camera", (1024, 1024), fov=(np.rad2deg(np.pi / 2), np.rad2deg(np.pi / 2))
    )
    scene = trimesh.Scene(
        [trimesh.points.PointCloud(v, colors)],
        camera=camera,
        camera_transform=cam2world,
    )
    scene.show()


# ── multi-env grid ─────────────────────────────────────────────────────────
def show_eval_grid(env, n_rows: int = 2, n_cols: int = 2, title: str = "") -> None:
    """Render a 2-D grid of parallel environments from ``env.render_rgb_array()``.

    Args:
        env: A ManiSkill gymnasium.Env with num_envs >= n_rows * n_cols.
        n_rows: Grid rows.
        n_cols: Grid columns.
        title: Super-title for the grid.
    """
    rgbs = env.render_rgb_array()  # (B, H, W, C)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for i, ax in enumerate(axs.flatten()):
        if i < rgbs.shape[0]:
            ax.imshow(rgbs[i].cpu().numpy())
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ── video playback ─────────────────────────────────────────────────────────
def display_video(path: str, width: int = 640) -> Video:
    """Return an IPython Video widget for inline Jupyter/Colab playback.

    Args:
        path: Path to an .mp4 file.
        width: Display width in pixels.

    Returns:
        IPython.display.Video object (rendered automatically in Jupyter).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    return Video(path, embed=True, width=width)


# ── training curve ─────────────────────────────────────────────────────────
def plot_training_curve(log_dir: str) -> None:
    """Plot loss curve from TensorBoard event files using tensorboard.

    Args:
        log_dir: The log directory passed to SummaryWriter.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("[viz] TensorBoard not available. Run:  pip install tensorboard")
        return

    ea = EventAccumulator(log_dir)
    ea.Reload()

    if "train/loss" not in ea.scalars.Keys():
        print("[viz] No 'train/loss' scalar found in log dir.")
        return

    events = ea.scalars.Items("train/loss")
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(9, 4))
    plt.plot(steps, values)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Diffusion Policy Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
