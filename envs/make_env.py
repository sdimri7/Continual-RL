"""
Environment factory functions.

All environment creation is centralised here so the notebook never
needs to touch gymnasium / mani_skill imports directly.
"""

from __future__ import annotations

import gymnasium as gym

# register all ManiSkill tasks
import mani_skill.envs  # noqa: F401

from mani_skill.utils.wrappers import RecordEpisode


# ── defaults ───────────────────────────────────────────────────────────────
PUSHT_DEFAULTS = dict(
    env_id="PushT-v1",
    obs_mode="rgb",
    control_mode="pd_ee_delta_pos",
    reward_mode="dense",
    render_mode="rgb_array",
    enable_shadow=True,
)


# ── generic factory ────────────────────────────────────────────────────────
def make_env(
    env_id: str,
    num_envs: int = 1,
    obs_mode: str = "state",
    control_mode: str = "pd_joint_delta_pos",
    reward_mode: str = "dense",
    render_mode: str = "rgb_array",
    record_dir: str | None = None,
    max_steps_per_video: int = 100,
    **kwargs,
) -> gym.Env:
    """Create a ManiSkill environment, optionally wrapping it with RecordEpisode.

    Args:
        env_id: ManiSkill task ID (e.g. "PushT-v1", "PickCube-v1").
        num_envs: Number of parallel environments (>1 enables GPU sim).
        obs_mode: Observation mode – "state", "rgb", "rgbd", "pointcloud".
        control_mode: Controller type – "pd_ee_delta_pos", "pd_joint_delta_pos", …
        reward_mode: "dense" or "sparse".
        render_mode: "rgb_array" (headless) or "human" (requires display).
        record_dir: If given, wraps env with RecordEpisode and saves videos here.
        max_steps_per_video: Steps between video flushes when recording.
        **kwargs: Extra kwargs forwarded to gym.make (e.g. robot_uids).

    Returns:
        gymnasium.Env (possibly wrapped with RecordEpisode).
    """
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        reward_mode=reward_mode,
        render_mode=render_mode,
        **kwargs,
    )

    if record_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=record_dir,
            save_trajectory=False,
            max_steps_per_video=max_steps_per_video,
        )

    return env


# ── Push-T convenience wrappers ────────────────────────────────────────────
def make_pusht_env(
    num_envs: int = 64,
    obs_mode: str = "rgb",
    record_dir: str | None = None,
    **overrides,
) -> gym.Env:
    """Create a Push-T training environment with sensible defaults.

    Args:
        num_envs: Parallel environments (64 fits comfortably on a Colab T4).
        obs_mode: "rgb" for image-based Diffusion Policy, "state" for MLP.
        record_dir: Optional path to save rollout videos.
        **overrides: Override any PUSHT_DEFAULTS key.
    """
    cfg = {**PUSHT_DEFAULTS, **overrides}
    return make_env(
        env_id=cfg["env_id"],
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=cfg["control_mode"],
        reward_mode=cfg["reward_mode"],
        render_mode=cfg["render_mode"],
        enable_shadow=cfg["enable_shadow"],
        record_dir=record_dir,
    )


def make_eval_env(
    num_envs: int = 10,
    obs_mode: str = "rgb",
    record_dir: str | None = None,
) -> gym.Env:
    """Single-task evaluation environment (fewer envs, always records video).

    Args:
        num_envs: Small count for deterministic evaluation rollouts.
        obs_mode: Must match the observation mode the policy was trained with.
        record_dir: Directory to save evaluation videos.
    """
    return make_pusht_env(
        num_envs=num_envs,
        obs_mode=obs_mode,
        record_dir=record_dir,
    )
