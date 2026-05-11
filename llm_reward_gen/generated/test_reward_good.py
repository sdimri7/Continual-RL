"""Test: a well-formed reward function that should pass all validation checks."""
import torch


def compute_dense_reward(self, obs, action, info):
    # Rotation reward: cos similarity between T-block and goal Z-rotation
    tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
    rot_rew = (tee_z_eulers - self.goal_z_rot).cos()
    reward = (((rot_rew + 1) / 2) ** 2) / 2

    # Position reward: tanh distance penalty in XY plane
    tee_to_goal = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
    dist = torch.linalg.norm(tee_to_goal, dim=1)
    reward += ((1 - torch.tanh(5 * dist)) ** 2) / 2

    # TCP proximity: encourage end-effector near T-block
    tcp_to_tee = self.tee.pose.p - self.agent.tcp.pose.p
    tcp_dist = torch.linalg.norm(tcp_to_tee, dim=1)
    reward += ((1 - torch.tanh(5 * tcp_dist)).sqrt()) / 20

    reward[info["success"]] = 3
    return reward
