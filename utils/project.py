"""
Project directory detection utilities.

Provides a centralized way to get the project root directory, eliminating
hardcoded paths from throughout the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> str:
    """
    Detect the project root directory.
    
    Traverses up from the current file to find the directory containing
    runner.py (which marks the project root).
    
    Returns:
        Absolute path to the project root directory.
    """
    # Start from this file's directory and traverse up
    current = Path(__file__).parent  # utils/
    project_root = current.parent    # Continual-RL/
    
    # Verify this is the project root by checking for runner.py
    if (project_root / "runner.py").exists():
        return str(project_root)
    
    # Fallback: use the directory containing this file's parent
    return str(project_root)


def get_demo_dir(env_id: str = "PushT-v1") -> str:
    """Get the demo directory for a given environment."""
    return os.path.join(get_project_root(), "demos", env_id)


def get_normalizer_path() -> str:
    """Get the path to the normalizer stats file."""
    return os.path.join(get_project_root(), "normalizer_stats.npz")


def get_checkpoint_dir() -> str:
    """Get the checkpoint directory."""
    return os.path.join(get_project_root(), "checkpoints")


def get_eval_video_dir() -> str:
    """Get the evaluation videos directory."""
    return os.path.join(get_project_root(), "eval_videos")


def get_runs_dir() -> str:
    """Get the training runs directory."""
    return os.path.join(get_project_root(), "runs")


def get_config_path() -> str:
    """Get the path to the config.json file."""
    return os.path.join(get_project_root(), "config.json")