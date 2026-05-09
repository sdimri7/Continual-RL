"""
Demo downloading, loading, and replay utilities.

All IO uses project root-relative paths via utils/project.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from mani_skill.utils.io_utils import load_json
from mani_skill.trajectory.utils import dict_to_list_of_dicts


# ── constants ──────────────────────────────────────────────────────────────
PUSHT_ENV_ID = "PushT-v1"


def _get_demos_root() -> str:
    """Get the demos directory, using project root detection."""
    from utils.project import get_project_root
    return os.path.join(get_project_root(), "demos")


# ── download ───────────────────────────────────────────────────────────────
def download_demos(
    env_id: str = PUSHT_ENV_ID,
    output_dir: str = None,
    force: bool = False,
) -> str:
    """Download ManiSkill demonstration data for an environment.

    Args:
        env_id: ManiSkill task ID.
        output_dir: Root directory under which demos are saved. 
                   Defaults to project demos directory.
        force: Re-download even if the directory already exists.

    Returns:
        Path to the demo directory for ``env_id``.
    """
    if output_dir is None:
        output_dir = _get_demos_root()
    demo_dir = os.path.join(output_dir, env_id)
    traj_file = os.path.join(demo_dir, "rl", "trajectory.none.pd_ee_delta_pos.physx_cuda.h5")

    if not force and os.path.exists(traj_file):
        print(f"[data] Demos already present at {demo_dir}. Skipping download.")
        return demo_dir

    os.makedirs(output_dir, exist_ok=True)
    cmd = f'python -m mani_skill.utils.download_demo "{env_id}" -o "{output_dir}"'
    print(f"[data] Downloading demos for {env_id} …")
    subprocess.run(cmd, shell=True, check=True)
    print(f"[data] Demos saved to {demo_dir}.")
    return demo_dir


def convert_demos(
    traj_path: str,
    obs_mode: str = "rgb",
    control_mode: str = "pd_ee_delta_pos",
    num_procs: int = 4,
    count: int | None = None,
) -> str:
    """Convert raw trajectories to a target observation / action space.

    The converted file is saved alongside the original .h5.

    Args:
        traj_path: Path to the original trajectory.h5.
        obs_mode: Target observation mode (e.g. "rgb", "rgbd").
        control_mode: Target controller.
        num_procs: Parallel conversion workers.
        count: Limit conversion to the first N trajectories (None = all).

    Returns:
        Path to the converted .h5 file.
    """
    count_flag = f"--count {count}" if count is not None else ""
    cmd = (
        f'python -m mani_skill.trajectory.replay_trajectory '
        f'--traj-path "{traj_path}" --save-traj '
        f'--obs-mode {obs_mode} -c "{control_mode}" '
        f'{count_flag}'
    )
    print(f"[data] Converting demos (obs_mode={obs_mode}, control_mode={control_mode}) …")
    subprocess.run(cmd, shell=True, check=True)

    # ManiSkill naming convention: trajectory.{obs_mode}.{control_mode}.physx_cuda.h5
    # e.g., trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5
    # Input: trajectory.none.pd_ee_delta_pos.physx_cuda.h5 -> Output: trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5
    out = traj_path.replace("trajectory.none.", f"trajectory.{obs_mode}.")
    print(f"[data] Converted demos saved to {out}.")
    return out


# ── loading ────────────────────────────────────────────────────────────────
def load_demo_metadata(traj_path: str) -> tuple[h5py.File, dict]:
    """Open a trajectory .h5 file and its companion .json metadata.

    Args:
        traj_path: Path to the .h5 file.

    Returns:
        Tuple of (h5py.File, json_data dict).
    """
    h5_file = h5py.File(traj_path, "r")
    json_data = load_json(traj_path.replace(".h5", ".json"))
    return h5_file, json_data


def print_h5_structure(h5_node: h5py.Group | h5py.File, prefix: str = "") -> None:
    """Recursively print the shape/dtype of every dataset in an HDF5 file."""
    for key in h5_node:
        if isinstance(h5_node[key], h5py.Group):
            print_h5_structure(h5_node[key], prefix=f"{prefix}/{key}")
        else:
            ds = h5_node[key]
            print(f"{prefix}/{key}  shape={ds.shape}  dtype={ds.dtype}")


# ── replay ─────────────────────────────────────────────────────────────────
def replay_episode(
    episode_idx: int,
    h5_file: h5py.File,
    json_data: dict,
    save_dir: str = "/tmp/replays",
    fps: int = 30,
) -> str:
    """Replay a single demonstration episode and save as an .mp4.

    Args:
        episode_idx: Index into json_data["episodes"].
        h5_file: Open h5py.File returned by load_demo_metadata.
        json_data: JSON metadata dict returned by load_demo_metadata.
        save_dir: Directory to write the replay video.
        fps: Output video frame-rate.

    Returns:
        Path to the saved video.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.visualization.misc import images_to_video

    os.makedirs(save_dir, exist_ok=True)
    episodes = json_data["episodes"]
    ep = episodes[episode_idx]
    episode_id = ep["episode_id"]
    traj = h5_file[f"traj_{episode_id}"]
    env_states = dict_to_list_of_dicts(traj["env_states"])

    env_kwargs = json_data["env_info"]["env_kwargs"]
    env = gym.make(json_data["env_info"]["env_id"], **env_kwargs)
    reset_kwargs = {**ep["reset_kwargs"], "seed": ep["episode_seed"]}
    env.reset(**reset_kwargs)

    frames: list[np.ndarray] = [env.render_rgb_array()[0].numpy()]
    for i in range(len(traj["actions"])):
        action = traj["actions"][i]
        env.step(action)
        env.set_state_dict(env_states[i])
        frames.append(env.render_rgb_array()[0].numpy())

    env.close()
    out_path = os.path.join(save_dir, f"replay_ep{episode_idx}.mp4")
    images_to_video(frames, output_dir=save_dir, video_name=f"replay_ep{episode_idx}", fps=fps)
    print(f"[data] Replay saved to {out_path}.")
    return out_path
