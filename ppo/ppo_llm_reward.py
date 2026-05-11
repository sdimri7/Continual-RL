"""PPO training on PushT with LLM-generated reward and failure-biased episode sampling.

This script adapts ppo/ppo.py to use the PushT-LLMReward-v1 environment,
which wraps PushTEnv with a dense reward function and episode sampler generated
by the LLM pipeline in llm_reward_gen/.

Usage:
    # Generate reward + episode config first:
    python llm_reward_gen/run_generate.py \\
        --config llm_reward_gen/failure_configs/rotation_failure.json \\
        --video-path path/to/failure_video.mp4

    # Then run fine-tuning (on GPU machine):
    python ppo/ppo_llm_reward.py \\
        --env-id PushT-LLMReward-v1 \\
        --reward-code-path llm_reward_gen/generated/reward_rotation_failure_v001.py \\
        --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py \\
        --failure-mode rotation_failure \\
        --num-envs 512 \\
        --total-timesteps 5000000

    # Evaluate on the targeted failure regime:
    python ppo/ppo_llm_reward.py \\
        --evaluate \\
        --checkpoint runs/<run_name>/final_ckpt.pt \\
        --episode-config-path llm_reward_gen/generated/episode_config_rotation_failure_v001.py
"""

from collections import defaultdict
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
import llm_reward_gen.custom_envs  # registers PushT-LLMReward-v1  # noqa: F401

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill-LLMReward"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # LLM reward specific
    reward_code_path: Optional[str] = None
    """Path to LLM-generated reward function .py file"""
    episode_config_path: Optional[str] = None
    """Path to LLM-generated episode config .py file"""
    failure_mode: str = "unknown"
    """Name of the failure mode being targeted"""
    failure_bias_ratio: float = 0.7
    """Fraction of episodes biased toward failure regime during training"""
    eval_failure_bias_ratio: float = 1.0
    """Use 100% failure-biased episodes for targeted evaluation"""
    eval_nominal: bool = False
    """If True, evaluate on nominal PushT distribution (no bias) to check forgetting"""

    # Environment
    env_id: str = "PushT-LLMReward-v1"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    num_envs: int = 512
    num_eval_envs: int = 8
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 100
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: str = "pd_joint_delta_pos"

    # PPO hyperparameters
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # Runtime (filled in main)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, act_dim) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def make_env(
    env_id,
    seed,
    reward_code_path=None,
    episode_config_path=None,
    failure_bias_ratio=0.7,
    video_dir=None,
    num_envs=1,
    reconfiguration_freq=None,
    control_mode="pd_joint_delta_pos",
):
    env_kwargs = dict(
        obs_mode="state",
        reward_mode="dense",
        control_mode=control_mode,
        sim_backend="physx_cuda",
        render_mode="cameras" if video_dir else None,
        reward_code_path=reward_code_path,
        episode_config_code_path=episode_config_path,
        failure_bias_ratio=failure_bias_ratio,
    )
    env = gym.make(
        env_id,
        num_envs=num_envs,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if video_dir:
        env = RecordEpisode(
            env,
            output_dir=video_dir,
            save_trajectory=False,
            max_steps_per_video=200,
            video_fps=30,
        )
    env = ManiSkillVectorEnv(
        env,
        num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )
    return env


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = f"ppo_llm_{args.failure_mode}"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Training envs: failure-biased episodes + LLM reward
    eval_video_dir = f"runs/{run_name}/eval_videos" if args.capture_video else None
    envs = make_env(
        args.env_id,
        seed=args.seed,
        reward_code_path=args.reward_code_path,
        episode_config_path=args.episode_config_path,
        failure_bias_ratio=args.failure_bias_ratio,
        num_envs=args.num_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        control_mode=args.control_mode,
    )

    # Evaluation envs: 100% failure-biased (targeted evaluation)
    eval_envs = make_env(
        args.env_id,
        seed=args.seed,
        reward_code_path=args.reward_code_path,
        episode_config_path=args.episode_config_path,
        failure_bias_ratio=args.eval_failure_bias_ratio,
        video_dir=eval_video_dir,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        control_mode=args.control_mode,
    )

    # Optional nominal evaluation (checks for catastrophic forgetting)
    nominal_eval_envs = None
    if args.eval_nominal:
        nominal_eval_envs = make_env(
            "PushT-v1",  # original env, no bias
            seed=args.seed + 100,
            num_envs=args.num_eval_envs,
            reconfiguration_freq=args.eval_reconfiguration_freq,
            control_mode=args.control_mode,
        )

    assert isinstance(envs.single_action_space, gym.spaces.Box)

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    writer = None
    if not args.evaluate:
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
        )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Storage
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions_buf = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    action_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(a):
        return torch.clamp(a.detach(), action_low, action_high)

    def run_eval(eval_env, obs_start, name_prefix):
        obs = obs_start
        metrics = defaultdict(list)
        num_episodes = 0
        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                obs, rew, term, trunc, infos = eval_env.step(
                    clip_action(agent.get_action(obs, deterministic=True))
                )
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].append(v)
        if writer:
            for k, v in metrics.items():
                mean_val = torch.stack(v).float().mean()
                writer.add_scalar(f"{name_prefix}/{k}", mean_val, global_step)
                print(f"  {name_prefix}_{k}: {mean_val:.4f}")
        return obs

    if args.evaluate:
        print("=== Evaluation mode ===")
        eval_obs = run_eval(eval_envs, eval_obs, "eval_targeted")
        if nominal_eval_envs:
            nom_obs, _ = nominal_eval_envs.reset(seed=args.seed + 200)
            run_eval(nominal_eval_envs, nom_obs, "eval_nominal")
        envs.close()
        eval_envs.close()
        if nominal_eval_envs:
            nominal_eval_envs.close()
        return

    print(f"Training for {args.num_iterations} iterations")
    for iteration in range(1, args.num_iterations + 1):
        print(f"Iter {iteration}/{args.num_iterations}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()

        if iteration % args.eval_freq == 1:
            eval_obs, _ = eval_envs.reset()
            eval_obs = run_eval(eval_envs, eval_obs, "eval_targeted")
            if nominal_eval_envs:
                nom_obs, _ = nominal_eval_envs.reset()
                run_eval(nominal_eval_envs, nom_obs, "eval_nominal")

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_time = time.time()
        agent.train()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                clip_action(action)
            )
            next_done = torch.logical_or(terminations, truncations).float()
            rewards_buf[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if writer:
                        writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = (
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)
                    )
        rollout_time = time.time() - rollout_time

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards_buf[t] + args.gamma * real_next_values - values_buf[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + values_buf

        # Flatten
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        if writer:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("time/rollout_time", rollout_time, global_step)
            writer.add_scalar("time/update_time", update_time, global_step)

        print(f"  SPS: {int(global_step / (time.time() - start_time))}")

    if args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if writer:
        writer.close()
    envs.close()
    eval_envs.close()
    if nominal_eval_envs:
        nominal_eval_envs.close()


if __name__ == "__main__":
    main()
