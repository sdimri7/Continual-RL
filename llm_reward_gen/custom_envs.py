"""Custom ManiSkill environments with LLM-generated rewards and episode configs."""

import importlib.util
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.push_t import PushTEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose


def _load_function_from_file(code_path: str, func_name: str):
    """Dynamically load a function from a Python file."""
    path = Path(code_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Generated code file not found: {path}")
    spec = importlib.util.spec_from_file_location("generated_module", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(
            f"Function '{func_name}' not found in {path}. "
            f"Available: {[x for x in dir(module) if not x.startswith('_')]}"
        )
    return fn


@register_env("PushT-LLMReward-v1", max_episode_steps=100)
class PushTLLMRewardEnv(PushTEnv):
    """PushT with LLM-generated dense reward and failure-biased episode initialization.

    Usage:
        import llm_reward_gen.custom_envs  # triggers registration

        env = gym.make(
            "PushT-LLMReward-v1",
            reward_code_path="llm_reward_gen/generated/reward_rotation_v1.py",
            episode_config_code_path="llm_reward_gen/generated/episode_rotation_v1.py",
            failure_bias_ratio=0.7,
            reward_mode="dense",
            obs_mode="state",
        )
    """

    def __init__(
        self,
        *args,
        reward_code_path: str = None,
        episode_config_code_path: str = None,
        failure_bias_ratio: float = 0.7,
        **kwargs,
    ):
        # Set these BEFORE super().__init__() because it calls reset() ->
        # _initialize_episode(), which reads self._episode_config_fn.
        self._reward_fn = None
        self._episode_config_fn = None
        self._failure_bias_ratio = failure_bias_ratio

        if reward_code_path is not None:
            self._reward_fn = _load_function_from_file(
                reward_code_path, "compute_dense_reward"
            )
        if episode_config_code_path is not None:
            self._episode_config_fn = _load_function_from_file(
                episode_config_code_path, "sample_failure_episode_config"
            )

        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs, action, info):
        if self._reward_fn is not None:
            return self._reward_fn(self, obs, action, info)
        return super().compute_dense_reward(obs, action, info)

    def compute_normalized_dense_reward(self, obs, action, info):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Set goal tee (fixed position, same as parent)
            target_region_xyz = torch.zeros((b, 3))
            target_region_xyz[:, 0] += self.goal_offset[0]
            target_region_xyz[:, 1] += self.goal_offset[1]
            target_region_xyz[..., 2] = 1e-3
            self.goal_tee.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, self.goal_z_rot),
                )
            )

            if self._episode_config_fn is not None:
                n_failure = int(b * self._failure_bias_ratio)
                n_uniform = b - n_failure

                all_pos_parts = []
                all_q_parts = []

                # Failure-biased samples
                if n_failure > 0:
                    fail_pos, fail_q = self._episode_config_fn(
                        n_failure, self.device
                    )
                    all_pos_parts.append(fail_pos)
                    all_q_parts.append(fail_q)

                # Uniform random samples (same as parent)
                if n_uniform > 0:
                    uni_xyz = torch.zeros((n_uniform, 3))
                    uni_xyz[:, 0] = self.goal_offset[0]
                    uni_xyz[:, 1] = self.goal_offset[1]
                    uni_xyz[..., 0] += (
                        torch.rand(n_uniform) * self.tee_spawnbox_xlength
                        + self.tee_spawnbox_xoffset
                    )
                    uni_xyz[..., 1] += (
                        torch.rand(n_uniform) * self.tee_spawnbox_ylength
                        + self.tee_spawnbox_yoffset
                    )
                    uni_xyz[..., 2] = 0.04 / 2 + 1e-3

                    q_euler = torch.rand(n_uniform) * (2 * torch.pi)
                    uni_q = torch.zeros((n_uniform, 4))
                    uni_q[:, 0] = (q_euler / 2).cos()
                    uni_q[:, -1] = (q_euler / 2).sin()

                    all_pos_parts.append(uni_xyz)
                    all_q_parts.append(uni_q)

                all_pos = torch.cat(all_pos_parts, dim=0)
                all_q = torch.cat(all_q_parts, dim=0)

                # Shuffle to avoid systematic ordering
                perm = torch.randperm(b)
                all_pos = all_pos[perm]
                all_q = all_q[perm]

                self.tee.set_pose(Pose.create_from_pq(p=all_pos, q=all_q))
            else:
                # Fall back to parent randomization logic
                xyz = target_region_xyz.clone()
                xyz[..., 0] += (
                    torch.rand(b) * self.tee_spawnbox_xlength
                    + self.tee_spawnbox_xoffset
                )
                xyz[..., 1] += (
                    torch.rand(b) * self.tee_spawnbox_ylength
                    + self.tee_spawnbox_yoffset
                )
                xyz[..., 2] = 0.04 / 2 + 1e-3

                q_euler_angle = torch.rand(b) * (2 * torch.pi)
                q = torch.zeros((b, 4))
                q[:, 0] = (q_euler_angle / 2).cos()
                q[:, -1] = (q_euler_angle / 2).sin()

                self.tee.set_pose(Pose.create_from_pq(p=xyz, q=q))

            # Set ee goal position marker (same as parent)
            ee_xyz = torch.zeros((b, 3))
            ee_xyz[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(
                Pose.create_from_pq(
                    p=ee_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
