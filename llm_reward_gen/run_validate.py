"""CLI entry point: validate LLM-generated reward function and episode config.

Usage:
    python llm_reward_gen/run_validate.py \\
        --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py \\
        --episode-config llm_reward_gen/generated/episode_config_rotation_failure_v001.py

    # Reward only:
    python llm_reward_gen/run_validate.py \\
        --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py

    # With full simulator (slower, more accurate):
    python llm_reward_gen/run_validate.py \\
        --reward-code llm_reward_gen/generated/reward_rotation_failure_v001.py \\
        --use-sim
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_reward_gen.validation import validate_episode_config, validate_reward_function


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate LLM-generated reward functions and episode configs."
    )
    parser.add_argument(
        "--reward-code",
        type=str,
        default=None,
        help="Path to generated reward function .py file",
    )
    parser.add_argument(
        "--episode-config",
        type=str,
        default=None,
        help="Path to generated episode config .py file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for tensor operations (default: cpu)",
    )
    parser.add_argument(
        "--use-sim",
        action="store_true",
        help="Use full ManiSkill simulator for validation (slower, more accurate)",
    )
    return parser.parse_args()


def validate_with_sim(reward_code_path: str, episode_config_path: str = None):
    """Perform integration validation with the full ManiSkill simulator."""
    import gymnasium as gym
    import torch
    import llm_reward_gen.custom_envs  # noqa: F401

    print("  Running simulator integration test...")
    try:
        env = gym.make(
            "PushT-LLMReward-v1",
            num_envs=16,
            obs_mode="state",
            reward_mode="dense",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cpu",
            render_mode=None,
            reward_code_path=reward_code_path,
            episode_config_code_path=episode_config_path,
            failure_bias_ratio=0.7 if episode_config_path else 0.0,
        )
        obs, _ = env.reset()
        rewards_seen = []
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_seen.append(reward)
        env.close()

        rewards_all = torch.stack(rewards_seen) if hasattr(rewards_seen[0], "shape") else rewards_seen
        print(f"    Simulator test passed: 10 steps completed")
        print(f"    Reward stats: mean={float(sum(r.mean() for r in rewards_seen)/len(rewards_seen)):.3f}")
        return True, "Simulator integration test passed"
    except Exception as e:
        import traceback
        return False, f"Simulator test failed: {e}\n{traceback.format_exc()}"


def main():
    args = parse_args()

    if not args.reward_code and not args.episode_config:
        print("ERROR: At least one of --reward-code or --episode-config is required")
        sys.exit(1)

    all_passed = True

    print(f"\n{'='*60}")
    print(f"LLM Output Validation")
    print(f"{'='*60}\n")

    # Validate reward function
    if args.reward_code:
        code_path = Path(args.reward_code)
        if not code_path.exists():
            print(f"ERROR: Reward code file not found: {code_path}")
            sys.exit(1)

        print(f"[Reward Function] {code_path.name}")
        passed, message = validate_reward_function(str(code_path), device=args.device)
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Status : {status}")
        print(f"  Details: {message}\n")
        if not passed:
            all_passed = False

    # Validate episode config
    if args.episode_config:
        config_path = Path(args.episode_config)
        if not config_path.exists():
            print(f"ERROR: Episode config file not found: {config_path}")
            sys.exit(1)

        print(f"[Episode Config] {config_path.name}")
        passed, message = validate_episode_config(str(config_path), device=args.device)
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Status : {status}")
        print(f"  Details: {message}\n")
        if not passed:
            all_passed = False

    # Full simulator integration test
    if args.use_sim and args.reward_code:
        print("[Simulator Integration Test]")
        passed, message = validate_with_sim(args.reward_code, args.episode_config)
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Status : {status}")
        print(f"  Details: {message}\n")
        if not passed:
            all_passed = False

    print(f"{'='*60}")
    overall = "✓ ALL CHECKS PASSED" if all_passed else "✗ SOME CHECKS FAILED"
    print(f"Overall: {overall}")
    print(f"{'='*60}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
