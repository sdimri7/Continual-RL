"""Test: a broken reward function that should FAIL validation checks."""
import torch


def compute_dense_reward(self, obs, action, info):
    # BAD: constant reward - no gradient signal
    B = self.tee.pose.p.shape[0]
    reward = torch.ones(B, device=self.device) * 0.5
    # Missing success bonus
    return reward
