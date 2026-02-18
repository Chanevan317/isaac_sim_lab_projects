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


def progress_reward(env: ManagerBasedRLEnv, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for moving towards the target (velocity projection)."""
    # 1. Get the direction vector from robot to target (World Frame)
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :2]
    to_target = target_pos - robot_pos
    
    # 2. Normalize the vector to get the 'heading' to the target
    to_target_dir = torch.nn.functional.normalize(to_target, dim=-1)
    
    # 3. Get the robot's current linear velocity in World Frame
    # (Using only X and Y since the cone is on the ground)
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    
    # 4. Calculate the Dot Product: (Velocity) ⋅ (Direction to Target)
    # This gives a positive value if moving towards, negative if moving away
    progress = torch.sum(vel_w * to_target_dir, dim=-1)
    
    # Optional: Clip the reward so it doesn't become too massive during high-speed resets
    return torch.clamp(progress, min=-1.0, max=1.0)


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

