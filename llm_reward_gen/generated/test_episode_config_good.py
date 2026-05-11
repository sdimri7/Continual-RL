"""Test: a well-formed episode config for rotation_failure that should pass validation."""
import torch


def sample_failure_episode_config(b: int, device: torch.device):
    """Bias toward initial Z-rotation in [1.5, 3.5] rad (rotation failure regime)."""
    goal_x, goal_y = -0.156, -0.1

    n_failure = int(b * 0.7)
    n_uniform = b - n_failure

    positions = torch.zeros(b, 3, device=device)
    quaternions = torch.zeros(b, 4, device=device)

    # Failure-biased: rotations in [1.5, 3.5] rad, positions near goal
    if n_failure > 0:
        fail_x = goal_x + (torch.rand(n_failure, device=device) * 0.2 - 0.1)
        fail_y = goal_y + (torch.rand(n_failure, device=device) * 0.3 - 0.1)
        positions[:n_failure, 0] = fail_x
        positions[:n_failure, 1] = fail_y
        positions[:n_failure, 2] = 0.021
        # Sample rotations in failure regime: [1.5, 3.5] rad
        fail_angles = torch.rand(n_failure, device=device) * 2.0 + 1.5
        quaternions[:n_failure, 0] = (fail_angles / 2).cos()  # w
        quaternions[:n_failure, 3] = (fail_angles / 2).sin()  # z

    # Uniform samples
    if n_uniform > 0:
        uni_x = goal_x + (torch.rand(n_uniform, device=device) * 0.2 - 0.1)
        uni_y = goal_y + (torch.rand(n_uniform, device=device) * 0.3 - 0.1)
        positions[n_failure:, 0] = uni_x
        positions[n_failure:, 1] = uni_y
        positions[n_failure:, 2] = 0.021
        uni_angles = torch.rand(n_uniform, device=device) * (2 * torch.pi)
        quaternions[n_failure:, 0] = (uni_angles / 2).cos()
        quaternions[n_failure:, 3] = (uni_angles / 2).sin()

    return positions, quaternions
