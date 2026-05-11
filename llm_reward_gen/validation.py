"""Validation checks for LLM-generated reward functions and episode configs."""

import importlib.util
import traceback
from pathlib import Path

import numpy as np
import torch


def _load_function(code_path: str, func_name: str):
    """Load a function from a generated .py file."""
    spec = importlib.util.spec_from_file_location("gen_module", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


class MockEnv:
    """Lightweight mock of PushTEnv for reward validation without full simulator."""

    def __init__(self, batch_size: int = 64, device: str = "cpu"):
        self.device = torch.device(device)
        self.b = batch_size
        self.goal_z_rot = (5 / 3) * np.pi

        class MockPose:
            def __init__(self, p, q):
                self.p = p
                self.q = q
                self.raw_pose = torch.cat([p, q], dim=-1)

        class MockActor:
            def __init__(self, pose):
                self.pose = pose

        class MockTCP:
            def __init__(self, pose):
                self.pose = pose

        class MockAgent:
            def __init__(self, tcp):
                self.tcp = tcp

        # Default: T-block at random positions, goal at fixed position
        tee_pos = torch.rand(batch_size, 3, device=self.device) * 0.3 - 0.3
        tee_pos[:, 2] = 0.021
        tee_angles = torch.rand(batch_size, device=self.device) * 2 * np.pi
        tee_q = torch.zeros(batch_size, 4, device=self.device)
        tee_q[:, 0] = (tee_angles / 2).cos()
        tee_q[:, -1] = (tee_angles / 2).sin()

        goal_pos = torch.zeros(batch_size, 3, device=self.device)
        goal_pos[:, 0] = -0.156
        goal_pos[:, 1] = -0.1
        goal_pos[:, 2] = 1e-3
        goal_q = torch.zeros(batch_size, 4, device=self.device)
        goal_q[:, 0] = (torch.tensor(self.goal_z_rot / 2)).cos()
        goal_q[:, -1] = (torch.tensor(self.goal_z_rot / 2)).sin()

        tcp_pos = torch.rand(batch_size, 3, device=self.device) * 0.3 - 0.3
        tcp_pos[:, 2] = 0.024
        tcp_q = torch.zeros(batch_size, 4, device=self.device)
        tcp_q[:, 0] = 1.0

        self.tee = MockActor(MockPose(tee_pos, tee_q))
        self.goal_tee = MockActor(MockPose(goal_pos, goal_q))
        self.agent = MockAgent(MockTCP(MockPose(tcp_pos, tcp_q)))

    def quat_to_z_euler(self, quats):
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        return 2 * qw.acos()

    def set_tee_pose(self, positions, quaternions):
        self.tee.pose.p = positions
        self.tee.pose.q = quaternions
        self.tee.pose.raw_pose = torch.cat([positions, quaternions], dim=-1)


def validate_reward_function(code_path: str, device: str = "cpu") -> tuple[bool, str]:
    """Validate a generated reward function.

    Returns:
        (passed: bool, message: str)
    """
    errors = []

    # Check 1: Syntax / import
    try:
        reward_fn = _load_function(code_path, "compute_dense_reward")
    except Exception as e:
        return False, f"Import error: {e}\n{traceback.format_exc()}"

    mock = MockEnv(batch_size=64, device=device)

    # Check 2: Execution and shape
    info = {"success": torch.zeros(64, dtype=torch.bool, device=mock.device)}
    obs = None
    action = torch.zeros(64, 7, device=mock.device)
    try:
        reward = reward_fn(mock, obs, action, info)
    except Exception as e:
        return False, f"Runtime error: {e}\n{traceback.format_exc()}"

    if not isinstance(reward, torch.Tensor):
        return False, f"Return type is {type(reward)}, expected torch.Tensor"
    if reward.shape != (64,):
        return False, f"Return shape is {reward.shape}, expected (64,)"

    # Check 3: Range
    if reward.isnan().any():
        errors.append("Reward contains NaN values")
    if reward.isinf().any():
        errors.append("Reward contains Inf values")
    if reward.min() < -10:
        errors.append(f"Reward min={reward.min():.3f} is below -10")
    if reward.max() > 10:
        errors.append(f"Reward max={reward.max():.3f} exceeds 10 (excluding success)")

    # Check 4: Success consistency
    info_success = {
        "success": torch.ones(64, dtype=torch.bool, device=mock.device)
    }
    try:
        reward_at_success = reward_fn(mock, obs, action, info_success)
        if not torch.allclose(
            reward_at_success, torch.tensor(3.0, device=mock.device), atol=0.1
        ):
            errors.append(
                f"Reward at success should be ~3.0, got {reward_at_success.mean():.3f}"
            )
    except Exception as e:
        errors.append(f"Error computing reward at success: {e}")

    # Check 5: Gradient signal — reward should correlate with proximity to goal
    mock_gradient = MockEnv(batch_size=100, device=device)
    goal_pos = mock_gradient.goal_tee.pose.p[0, :2]
    goal_z = mock_gradient.goal_z_rot

    # Interpolate positions from far to goal
    t = torch.linspace(0, 1, 100, device=mock_gradient.device).unsqueeze(1)
    start_pos = torch.tensor([[-0.3, 0.2]], device=mock_gradient.device)
    interp_pos = start_pos * (1 - t) + goal_pos.unsqueeze(0) * t
    full_pos = torch.zeros(100, 3, device=mock_gradient.device)
    full_pos[:, :2] = interp_pos
    full_pos[:, 2] = 0.021

    # Set rotation to goal rotation
    goal_q = torch.zeros(100, 4, device=mock_gradient.device)
    goal_q[:, 0] = (torch.tensor(goal_z / 2)).cos()
    goal_q[:, -1] = (torch.tensor(goal_z / 2)).sin()

    mock_gradient.set_tee_pose(full_pos, goal_q)

    info_no_success = {
        "success": torch.zeros(100, dtype=torch.bool, device=mock_gradient.device)
    }
    try:
        rewards_interp = reward_fn(
            mock_gradient, None, torch.zeros(100, 7, device=mock_gradient.device),
            info_no_success,
        )
        # Check correlation: rewards should generally increase as we approach goal
        dists = torch.linalg.norm(full_pos[:, :2] - goal_pos, dim=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(
                -dists.cpu().numpy(), rewards_interp.detach().cpu().numpy()
            )[0, 1]
        if np.isnan(corr):
            corr = 0.0  # constant reward → zero correlation
        if corr < 0.3:
            errors.append(
                f"Low correlation ({corr:.3f}) between proximity-to-goal and reward. "
                f"Reward should increase as T-block approaches goal."
            )
    except Exception as e:
        errors.append(f"Gradient signal check failed: {e}")

    if errors:
        return False, "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
    return True, "All checks passed"


def validate_episode_config(
    code_path: str, device: str = "cpu"
) -> tuple[bool, str]:
    """Validate a generated episode configuration sampler.

    Returns:
        (passed: bool, message: str)
    """
    errors = []

    # Check 1: Import
    try:
        config_fn = _load_function(code_path, "sample_failure_episode_config")
    except Exception as e:
        return False, f"Import error: {e}\n{traceback.format_exc()}"

    # Check 2: Execution and shapes
    device_obj = torch.device(device)
    try:
        positions, quaternions = config_fn(256, device_obj)
    except Exception as e:
        return False, f"Runtime error: {e}\n{traceback.format_exc()}"

    if not isinstance(positions, torch.Tensor):
        return False, f"positions type is {type(positions)}, expected torch.Tensor"
    if not isinstance(quaternions, torch.Tensor):
        return False, f"quaternions type is {type(quaternions)}, expected torch.Tensor"
    if positions.shape != (256, 3):
        return False, f"positions shape is {positions.shape}, expected (256, 3)"
    if quaternions.shape != (256, 4):
        return False, f"quaternions shape is {quaternions.shape}, expected (256, 4)"

    # Check 3: Position bounds
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    z_vals = positions[:, 2]

    if x_vals.min() < -0.5 or x_vals.max() > 0.1:
        errors.append(
            f"x positions out of bounds: [{x_vals.min():.3f}, {x_vals.max():.3f}], "
            f"expected [-0.5, 0.1]"
        )
    if y_vals.min() < -0.5 or y_vals.max() > 0.4:
        errors.append(
            f"y positions out of bounds: [{y_vals.min():.3f}, {y_vals.max():.3f}], "
            f"expected [-0.5, 0.4]"
        )
    z_target = 0.021
    if not torch.allclose(z_vals, torch.tensor(z_target, device=device_obj), atol=0.005):
        errors.append(
            f"z positions should be ~{z_target}, got range "
            f"[{z_vals.min():.4f}, {z_vals.max():.4f}]"
        )

    # Check 4: Quaternion validity
    quat_norms = torch.linalg.norm(quaternions, dim=1)
    if not torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=0.01):
        errors.append(
            f"Quaternions not unit norm: range [{quat_norms.min():.4f}, {quat_norms.max():.4f}]"
        )

    # qx and qy should be 0 (Z-rotation only)
    qx_max = quaternions[:, 1].abs().max()
    qy_max = quaternions[:, 2].abs().max()
    if qx_max > 0.01 or qy_max > 0.01:
        errors.append(
            f"Quaternions have non-zero qx/qy (max qx={qx_max:.4f}, qy={qy_max:.4f}). "
            f"Only Z-rotation is expected."
        )

    # Check 5: NaN / Inf
    if positions.isnan().any() or quaternions.isnan().any():
        errors.append("Output contains NaN values")
    if positions.isinf().any() or quaternions.isinf().any():
        errors.append("Output contains Inf values")

    if errors:
        return False, "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
    return True, "All checks passed"
