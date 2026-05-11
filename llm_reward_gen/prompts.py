"""Prompt templates for LLM-driven reward and episode configuration generation."""

REWARD_SYSTEM_PROMPT = """\
You are a robotics reward engineer for reinforcement learning. You write dense \
reward functions as executable Python code that operates on a physics simulator's \
internal state. You are precise about tensor shapes and always produce vectorized, \
GPU-compatible PyTorch code."""

REWARD_USER_PROMPT_TEMPLATE = """\
## Task
A robot arm with a stick end-effector pushes a T-shaped block to match a goal T \
position and orientation on a table. Success = T-block covers >=90% of goal T area.

## Environment API (ManiSkill PushTEnv)
Your reward function is a method `compute_dense_reward(self, obs, action, info)`.
It has access to:

  self.tee.pose.p           # T-block position, shape (B, 3), [x, y, z]
  self.tee.pose.q           # T-block quaternion, shape (B, 4), [w, x, y, z]
  self.goal_tee.pose.p      # Goal T position, shape (B, 3), fixed [-0.156, -0.1, 0.001]
  self.goal_z_rot           # Goal Z-rotation, scalar = 5.236 rad ((5/3)*pi)
  self.agent.tcp.pose.p     # End-effector position, shape (B, 3)
  self.quat_to_z_euler(q)   # Quaternion batch -> Z euler angles, (B,4) -> (B,)
  info["success"]           # Boolean tensor, shape (B,), True if >=90% overlap
  torch                     # PyTorch is available; all tensors on self.device

Workspace bounds: x in [-0.5, 0.1], y in [-0.4, 0.3], z ~ 0.02 (table surface)
T-block dimensions: horizontal bar 0.2x0.05, vertical bar 0.05x0.15
Center of mass offset: (0, 0.0375) from horizontal bar center

## Current Baseline Reward (achieves ~{baseline_success_rate}% success rate)
```python
def compute_dense_reward(self, obs, action, info):
    tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
    rot_rew = (tee_z_eulers - self.goal_z_rot).cos()
    reward = (((rot_rew + 1) / 2) ** 2) / 2

    tee_to_goal_pose = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
    tee_to_goal_pose_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
    reward += ((1 - torch.tanh(5 * tee_to_goal_pose_dist)) ** 2) / 2

    tcp_to_push_pose = self.tee.pose.p - self.agent.tcp.pose.p
    tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
    reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

    reward[info["success"]] = 3
    return reward
```

## Failure Mode: {failure_mode_name}
{failure_mode_description}

Quantitative characterization:
{quantitative_chars}

## Attached: {num_frames} frames from a failed episode showing this failure mode

## Your Task
Write an IMPROVED `compute_dense_reward` function that specifically addresses this \
failure mode. Requirements:
1. Return a torch.Tensor of shape (B,) where B = batch size
2. Keep reward[info["success"]] = 3 for consistency
3. Each component should be smooth (no hard if/else on continuous variables)
4. Each component should be in [0, 1] range before summing
5. Total non-success reward should be in [0, 2] range
6. Add reward terms that specifically incentivize recovery from this failure
7. Do NOT call self.pseudo_render_intersection() -- it is too expensive

Output ONLY the Python function in ```python fences. No explanation outside code."""


EPISODE_CONFIG_SYSTEM_PROMPT = """\
You are a robotics curriculum designer for reinforcement learning. You generate \
episode initialization configurations that bias training toward specific failure \
regimes. You write vectorized PyTorch code."""

EPISODE_CONFIG_USER_PROMPT_TEMPLATE = """\
## Task
A robot arm with a stick end-effector pushes a T-shaped block to match a goal T \
position and orientation on a table. Success = T-block covers >=90% of goal T area.

## Current Episode Initialization
T-block spawns randomly:
  x: goal_x + uniform(-0.1, 0.1)     where goal_x = -0.156
  y: goal_y + uniform(-0.1, 0.2)     where goal_y = -0.1
  z: 0.021 (fixed, on table surface)
  Z-rotation: uniform(0, 2*pi)

Goal is fixed at position [-0.156, -0.1] with Z-rotation (5/3)*pi ~ 5.236 rad.

## Failure Mode: {failure_mode_name}
{failure_mode_description}

Quantitative characterization:
{quantitative_chars}

## Your Task
Write a function that samples T-block initial configurations biased toward this \
failure regime:

```python
def sample_failure_episode_config(b: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Args:
        b: batch size (number of parallel environments)
        device: torch device
    Returns:
        positions: (b, 3) tensor of T-block initial positions
        quaternions: (b, 4) tensor of T-block initial orientations [w, x, y, z]
    \"\"\"
```

Requirements:
1. ~70% of samples should be in the failure regime, ~30% uniform random (for diversity)
2. Positions must be within reachable workspace: x in [-0.356, -0.056], y in [-0.3, 0.2]
3. z position must be 0.021
4. Quaternions must be unit-norm, Z-rotation only (qx=0, qy=0)
5. Use torch operations only (no numpy), all tensors on the given device

Output ONLY the Python function in ```python fences. No explanation outside code."""


REFINEMENT_PROMPT_TEMPLATE = """\
The previously generated {artifact_type} failed validation.

Error: {error_message}

Statistics: {statistics}

Here is the problematic code:
```python
{previous_code}
```

Please fix the function. Keep all the original requirements. \
Output ONLY the corrected Python function in ```python fences."""


def build_reward_prompt(
    failure_mode_name: str,
    failure_mode_description: str,
    quantitative_chars: list[str],
    baseline_success_rate: float = 60,
    num_frames: int = 8,
) -> str:
    chars_text = "\n".join(f"- {c}" for c in quantitative_chars)
    return REWARD_USER_PROMPT_TEMPLATE.format(
        failure_mode_name=failure_mode_name,
        failure_mode_description=failure_mode_description,
        quantitative_chars=chars_text,
        baseline_success_rate=baseline_success_rate,
        num_frames=num_frames,
    )


def build_episode_config_prompt(
    failure_mode_name: str,
    failure_mode_description: str,
    quantitative_chars: list[str],
) -> str:
    chars_text = "\n".join(f"- {c}" for c in quantitative_chars)
    return EPISODE_CONFIG_USER_PROMPT_TEMPLATE.format(
        failure_mode_name=failure_mode_name,
        failure_mode_description=failure_mode_description,
        quantitative_chars=chars_text,
    )


def build_refinement_prompt(
    artifact_type: str,
    error_message: str,
    statistics: str,
    previous_code: str,
) -> str:
    return REFINEMENT_PROMPT_TEMPLATE.format(
        artifact_type=artifact_type,
        error_message=error_message,
        statistics=statistics,
        previous_code=previous_code,
    )
