"""CLI entry point: generate LLM reward function and episode config from a failure video.

Usage:
    python llm_reward_gen/run_generate.py \\
        --video-path eval_videos/failure_rotation_001.mp4 \\
        --failure-mode rotation_failure \\
        --failure-description "Robot pushes T-block near goal position but fails to achieve correct orientation; block ends up rotated 60-120 degrees off target." \\
        --quantitative-chars "T-block reaches within 0.03m of goal but rotation error > 45deg" "Occurs when initial T-block Z-rotation is in [1.5, 3.5] rad" \\
        --baseline-success-rate 62 \\
        --model claude-sonnet-4-20250514

    # Or with a pre-written config file:
    python llm_reward_gen/run_generate.py --config llm_reward_gen/failure_configs/rotation_failure.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_reward_gen.pipeline import generate_episode_config, generate_reward_function
from llm_reward_gen.validation import validate_episode_config, validate_reward_function


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LLM reward function and episode config from a failure video."
    )
    parser.add_argument("--video-path", type=str, help="Path to failure mode mp4 video")
    parser.add_argument(
        "--failure-mode",
        type=str,
        help="Short snake_case name for the failure mode (e.g. rotation_failure)",
    )
    parser.add_argument(
        "--failure-description",
        type=str,
        help="Qualitative description of the failure mode",
    )
    parser.add_argument(
        "--quantitative-chars",
        nargs="+",
        default=[],
        help="Quantitative characterization strings (one or more)",
    )
    parser.add_argument(
        "--baseline-success-rate",
        type=float,
        default=60.0,
        help="Baseline policy success rate in percent (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of frames to extract from video (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum LLM refinement attempts (default: 3)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation checks (faster but less safe)",
    )
    parser.add_argument(
        "--reward-only",
        action="store_true",
        help="Only generate the reward function, skip episode config",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with all arguments (overrides CLI flags)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load from config file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = json.load(f)
        # Override args with config values
        for key, val in config.items():
            setattr(args, key.replace("-", "_"), val)

    # Validate required arguments
    if not args.video_path:
        print("ERROR: --video-path is required")
        sys.exit(1)
    if not args.failure_mode:
        print("ERROR: --failure-mode is required")
        sys.exit(1)
    if not args.failure_description:
        print("ERROR: --failure-description is required")
        sys.exit(1)

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"LLM Reward Generation Pipeline")
    print(f"{'='*60}")
    print(f"  Failure mode : {args.failure_mode}")
    print(f"  Video        : {video_path}")
    print(f"  Model        : {args.model}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Max retries  : {args.max_retries}")
    print(f"  Validation   : {'disabled' if args.skip_validation else 'enabled'}")
    print(f"{'='*60}\n")

    reward_validate_fn = None if args.skip_validation else (
        lambda p: validate_reward_function(p, device="cpu")
    )
    episode_validate_fn = None if args.skip_validation else (
        lambda p: validate_episode_config(p, device="cpu")
    )

    # Generate reward function
    print("[1/2] Generating reward function...")
    reward_path = generate_reward_function(
        video_path=str(video_path),
        failure_mode=args.failure_mode,
        failure_description=args.failure_description,
        quantitative_chars=args.quantitative_chars,
        baseline_success_rate=args.baseline_success_rate,
        model=args.model,
        temperature=args.temperature,
        num_frames=args.num_frames,
        max_retries=args.max_retries,
        validate_fn=reward_validate_fn,
    )
    print(f"    -> Saved to: {reward_path}\n")

    # Generate episode config
    if not args.reward_only:
        print("[2/2] Generating episode configuration sampler...")
        episode_path = generate_episode_config(
            video_path=str(video_path),
            failure_mode=args.failure_mode,
            failure_description=args.failure_description,
            quantitative_chars=args.quantitative_chars,
            model=args.model,
            temperature=args.temperature,
            num_frames=args.num_frames,
            max_retries=args.max_retries,
            validate_fn=episode_validate_fn,
        )
        print(f"    -> Saved to: {episode_path}\n")
    else:
        episode_path = None

    print(f"{'='*60}")
    print(f"Generation complete!")
    print(f"  Reward code  : {reward_path}")
    if episode_path:
        print(f"  Episode code : {episode_path}")
    print(f"\nTo train with these outputs:")
    print(f"  python ppo/ppo.py \\")
    print(f"    --env-id PushT-LLMReward-v1 \\")
    print(f"    --reward-code-path {reward_path} \\")
    if episode_path:
        print(f"    --episode-config-path {episode_path} \\")
    print(f"    --reward-mode dense")
    print(f"{'='*60}\n")

    # Print example env instantiation code
    print("Or instantiate directly:")
    print(f"""
import gymnasium as gym
import llm_reward_gen.custom_envs  # registers PushT-LLMReward-v1

env = gym.make(
    "PushT-LLMReward-v1",
    num_envs=512,
    obs_mode="state",
    reward_mode="dense",
    control_mode="pd_joint_delta_pos",
    sim_backend="physx_cuda",
    reward_code_path="{reward_path}",{"" if not episode_path else chr(10) + f'    episode_config_code_path="{episode_path}",' }
    failure_bias_ratio=0.7,
)
""")


if __name__ == "__main__":
    main()
