from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ict_bot.tasks.obstacle_avoidance import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reaching_target_reward(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg):
    # Distance between robot and cone
    robot_pos = env.scene["robot"].data.root_pos_w
    cone_pos = env.scene[target_cfg.name].data.root_pos_w
    distance = torch.norm(cone_pos - robot_pos, dim=-1)
    return 1.0 / (1.0 + distance) # Inverse distance reward

def obstacle_avoidance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float):
    """Penalize the robot if ANY ray detects an obstacle closer than the threshold."""
    # 1. Get the distances we calculated earlier (Shape: [128, 441])
    # Note: Make sure this calls your 'get_raycast_distances' function logic
    distances = mdp.get_raycast_distances(env, sensor_cfg)
    
    # 2. Check which rays are 'too close' (Shape: [128, 441])
    too_close = (distances < threshold)
    
    # 3. REDUCE: If ANY ray in an environment is too close, apply penalty
    # .any(dim=1) converts [128, 441] -> [128]
    penalty = torch.any(too_close, dim=1).float()
    
    # Now it returns shape (128,), which matches 'tensor a'
    return penalty


# def straightness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Penalize Y-axis velocity (lateral motion in world frame)."""
#     vel_y = env.scene["robot"].data.root_lin_vel_w[:, 1]
#     return torch.abs(vel_y)


def idle_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize if robot is not moving forward (low X velocity)."""
    vel_x = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return (torch.abs(vel_x) < 0.01).float()