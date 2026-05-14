#!/usr/bin/env python
"""
Run evaluations across multiple random seeds and compute the baseline average success rate.
Supports both Diffusion Policy and PPO (Standard/LLMReward).
"""

import argparse
import subprocess
import re
import sys
import os

def extract_success_rate(output, agent_type):
    """Extract the overall success rate from script stdout"""
    if agent_type == "diffusion":
        # Look for the final CSV or printed metric line: eval_success_once: X.XXX
        match = re.search(r'eval_success_once[ =:,]+([0-9.]+)', output)
        if match:
            return float(match.group(1))
    elif agent_type == "ppo":
        # Look for eval_success_once_mean=X.XXX from the PPO script log
        match = re.search(r'eval_success_once_mean\s*=\s*([0-9.]+)', output)
        if match:
            return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate over multiple seeds to establish baseline.")
    parser.add_argument("--agent-type", choices=["diffusion", "ppo"], required=True,
                        help="The type of agent to evaluate (diffusion or ppo)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--env-id", type=str, default="PushT-v1",
                        help="Environment ID (e.g., PushT-v1, PushT-LLMReward-v1)")
    parser.add_argument("--episode-config-path", type=str, default=None,
                        help="Path to episode config (needed for LLM Reward PPO envs)")
    parser.add_argument("--num-eval-envs", type=int, default=50,
                        help="Number of parallel evaluation environments")
    parser.add_argument("--num-eval-episodes", type=int, default=250,
                        help="Total number of episodes to evaluate per seed")
    parser.add_argument("--max-steps", type=int, default=150,
                        help="Maximum steps per episode")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose",
                        help="Control mode")
    parser.add_argument("--obs-mode", type=str, default="rgb",
                        help="Observation mode (used for diffusion)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of seeds to evaluate on")
    parser.add_argument("--capture-video", action="store_true", default=False,
                        help="Whether to capture video")
    parser.add_argument("--video-dir", type=str, default="eval_outputs/videos",
                        help="Base directory to save videos (will create seed_X subdirectories here)")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    print(f"Evaluating {args.agent_type.upper()} model across {len(args.seeds)} seeds:")
    print(f"Environment: {args.env_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes per seed: {args.num_eval_episodes}")
    print("=" * 60)

    success_rates = []

    for seed in args.seeds:
        print(f"\n--- Running evaluation for seed {seed} ---")

        if args.agent_type == "diffusion":
            cmd = [
                "python", "official_diffusion_policy/eval_rgbd.py",
                "--checkpoint", args.checkpoint,
                "--env-id", args.env_id,
                "--control-mode", args.control_mode,
                "--obs-mode", args.obs_mode,
                "--max_episode_steps", str(args.max_steps),
                "--num-eval-envs", str(args.num_eval_envs),
                "--num-eval-episodes", str(args.num_eval_episodes),
                "--sim-backend", "physx_cuda",
                "--seed", str(seed)
            ]
            if args.capture_video:
                cmd.extend(["--capture-video", "--video-dir", os.path.join(args.video_dir, f"seed_{seed}")])

        elif args.agent_type == "ppo":
            script_path = "ppo/ppo.py"
            if "LLMReward" in args.env_id:
                script_path = "ppo/ppo_llm_reward.py"

            cmd = [
                "python", script_path,
                "--evaluate",
                "--checkpoint", args.checkpoint,
                "--env-id", args.env_id,
                "--control-mode", args.control_mode,
                "--num-eval-envs", str(args.num_eval_envs),
                # PPO determines total evals using steps * envs
                "--num-eval-steps", str(args.max_steps * (args.num_eval_episodes // args.num_eval_envs + 2)),
                "--seed", str(seed)
            ]

            # For PPO scripts, video directory might need to be passed as an ENV var or
            # if we can't easily change the hardcoded directory, we'll note where they go
            if args.episode_config_path:
                cmd.extend(["--episode-config-path", args.episode_config_path])
            if args.capture_video:
                cmd.append("--capture-video")

        try:
            print(f"Exec: {' '.join(cmd)}")
            # Use subprocess and pipe output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            output_log = ""
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # print line as it comes in
                output_log += line

            process.stdout.close()
            return_code = process.wait()

            if return_code != 0:
                print(f"Warning: Script exited with error code {return_code} for seed {seed}")

            # Extract success rate
            rate = extract_success_rate(output_log, args.agent_type)
            if rate is not None:
                success_rates.append(rate)
                print(f"\n=> Scored {rate:.4f} Success Rate for Seed {seed}")
            else:
                print(f"\n=> Warning: Could not find success rate log from output for seed {seed}")

        except Exception as e:
            print(f"Run failed for seed {seed}: {e}")

    # Calculate and output summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if len(success_rates) > 0:
        baseline_avg = sum(success_rates) / len(success_rates)
        print(f"Checked checkpoint: {args.checkpoint}")
        print(f"Seeds checked:      {len(success_rates)} / {len(args.seeds)}")
        print(f"Individual rates:   {[f'{r:.4f}' for r in success_rates]}")
        print(f"\nFINAL BASELINE AVERAGE SUCCESS RATE: {baseline_avg:.4f}")
    else:
        print("Failed to record any success rates. Please check the logs.")

if __name__ == "__main__":
    main()