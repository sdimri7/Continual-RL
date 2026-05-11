"""Frame extraction and failure mode classification from evaluation rollout videos."""

import base64
import io
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path: str, num_frames: int = 8) -> list[np.ndarray]:
    """Extract N uniformly-spaced frames from an mp4 file.

    Args:
        video_path: Path to the mp4 video file.
        num_frames: Number of frames to extract.

    Returns:
        List of BGR numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Could not read video: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be extracted from {video_path}")
    return frames


def frames_to_base64(frames: list[np.ndarray]) -> list[str]:
    """Convert BGR frames to base64-encoded PNG strings for LLM vision APIs."""
    encoded = []
    for frame in frames:
        _, buffer = cv2.imencode(".png", frame)
        b64 = base64.b64encode(buffer).decode("utf-8")
        encoded.append(b64)
    return encoded


def create_frame_grid(frames: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Arrange frames into a single grid image for compact LLM input.

    Args:
        frames: List of BGR frames (must all be same size).
        cols: Number of columns in the grid.

    Returns:
        Single BGR image with frames arranged in a grid.
    """
    if not frames:
        raise ValueError("No frames to create grid from")

    h, w = frames[0].shape[:2]
    target_h, target_w = 256, 256
    resized = [cv2.resize(f, (target_w, target_h)) for f in frames]

    rows_needed = (len(resized) + cols - 1) // cols
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    row_images = []
    for r in range(rows_needed):
        row = np.hstack(resized[r * cols : (r + 1) * cols])
        row_images.append(row)
    grid = np.vstack(row_images)
    return grid


def extract_state_trajectory_from_h5(h5_path: str, traj_id: int) -> dict:
    """Load state trajectory from an evaluation HDF5 file.

    Args:
        h5_path: Path to the HDF5 file containing trajectories.
        traj_id: Index of the trajectory to load.

    Returns:
        Dict with 'obs' (T, obs_dim), 'actions' (T-1, act_dim), 'success' bool.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        traj_key = f"traj_{traj_id}"
        if traj_key not in f:
            raise KeyError(f"{traj_key} not found in {h5_path}")
        traj = f[traj_key]
        result = {
            "obs": np.array(traj["obs"]) if "obs" in traj else None,
            "actions": np.array(traj["actions"]) if "actions" in traj else None,
            "env_states": (
                np.array(traj["env_states"]) if "env_states" in traj else None
            ),
            "success": bool(traj.attrs.get("success", False)),
        }
    return result


def analyze_failure_from_states(
    obj_poses: np.ndarray,
    goal_pos: np.ndarray,
    goal_z_rot: float,
) -> dict:
    """Analyze a trajectory to classify the failure mode.

    Args:
        obj_poses: (T, 7) array of T-block poses [x, y, z, w, qx, qy, qz].
        goal_pos: (3,) goal position.
        goal_z_rot: Goal Z rotation in radians.

    Returns:
        Dict with 'failure_mode' string and supporting metrics.
    """
    final_pos = obj_poses[-1, :3]
    final_quat = obj_poses[-1, 3:]

    pos_dist = np.linalg.norm(final_pos[:2] - goal_pos[:2])

    qw = final_quat[0]
    sign = 1.0 if final_quat[-1] >= 0 else -1.0
    z_euler = 2 * np.arccos(np.clip(qw * sign, -1, 1))
    rot_error = abs(np.cos(z_euler - goal_z_rot) - 1.0)

    pos_trajectory = obj_poses[:, :2]
    displacements = np.linalg.norm(np.diff(pos_trajectory, axis=0), axis=1)
    total_movement = displacements.sum()

    dists_to_goal = np.linalg.norm(pos_trajectory - goal_pos[:2], axis=1)
    crossed_goal = np.any(dists_to_goal < 0.03) and dists_to_goal[-1] > 0.05

    metrics = {
        "final_pos_dist": float(pos_dist),
        "final_rot_error_cos": float(rot_error),
        "total_movement": float(total_movement),
        "crossed_goal": bool(crossed_goal),
        "final_z_euler": float(z_euler),
    }

    if pos_dist < 0.05 and rot_error > 0.3:
        metrics["failure_mode"] = "rotation_failure"
    elif crossed_goal:
        metrics["failure_mode"] = "overshoot"
    elif total_movement < 0.02:
        metrics["failure_mode"] = "stuck"
    elif pos_dist < 0.1 and rot_error < 0.5:
        metrics["failure_mode"] = "partial_alignment"
    else:
        metrics["failure_mode"] = "wrong_approach"

    return metrics
