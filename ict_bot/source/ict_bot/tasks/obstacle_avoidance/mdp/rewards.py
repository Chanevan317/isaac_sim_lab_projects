from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ict_bot.tasks.obstacle_avoidance import mdp
from isaaclab.utils.math import quat_apply

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
    # # 1. Get the direction vector from robot to target (World Frame)
    # robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    # target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :2]
    # to_target = target_pos - robot_pos
    
    # # 2. Normalize the vector to get the 'heading' to the target
    # to_target_dir = torch.nn.functional.normalize(to_target, dim=-1)
    
    # # 3. Get the robot's current linear velocity in World Frame
    # # (Using only X and Y since the cone is on the ground)
    # vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    
    # # 4. Calculate the Dot Product: (Velocity) ⋅ (Direction to Target)
    # # This gives a positive value if moving towards, negative if moving away
    # progress = torch.sum(vel_w * to_target_dir, dim=-1)
    
    # # Optional: Clip the reward so it doesn't become too massive during high-speed resets
    # return torch.clamp(progress, min=-1.0, max=1.0)

    # 1. Get Direction to Target (World XY)
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :2]
    to_target = target_pos - robot_pos
    to_target_dir = torch.nn.functional.normalize(to_target, dim=-1)
    
    # 2. Get Robot's Forward Vector (Local +X rotated to World)
    root_quat = env.scene["robot"].data.root_quat_w
    # NOTE: If your robot still considers 'back' as 'front', 
    # change [1.0, 0.0, 0.0] to [-1.0, 0.0, 0.0] below.
    forward_vec_template = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(root_quat.shape[0], 1)
    forward_vec_w = quat_apply(root_quat, forward_vec_template)[:, :2]
    forward_vec_w = torch.nn.functional.normalize(forward_vec_w, dim=-1)

    # 3. Calculate Heading Alignment (Cosine Similarity)
    # 1.0 = Facing target, -1.0 = Back to target
    heading_alignment = torch.sum(forward_vec_w * to_target_dir, dim=-1)

    # 4. Get Velocity Projection (Original Progress)
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    progress = torch.sum(vel_w * to_target_dir, dim=-1)
    
    # 5. MASKED REWARD: Only give progress if heading_alignment > 0 (Facing mostly toward)
    # This prevents the 'Driving Backward' strategy.
    facing_mask = (heading_alignment > 0.5).float() # Must be within ~60 degrees of target
    
    # Alternatively: Multiply progress by alignment so it scales
    # combined_reward = progress * torch.clamp(heading_alignment, min=0.0)
    
    final_reward = progress * facing_mask
    
    return torch.clamp(final_reward, min=-1.0, max=1.0)

def progress_reward_conditional(env, target_cfg):
    # Calculate Heading Alignment (Cosine Similarity)
    # alignment = 1.0 (perfectly facing), -1.0 (back to target)
    alignment = mdp.heading_to_target_reward(env, target_cfg)
    
    # Calculate Raw Velocity toward target
    vel_w = env.scene["robot"].data.root_lin_vel_w[:, :2]
    to_target_dir = mdp.get_direction_to_target(env, target_cfg)
    progress = torch.sum(vel_w * to_target_dir, dim=-1)
    
    # Only reward progress if alignment > 0.7 (~45 degrees)
    facing_mask = (alignment > 0.7).float()
    
    return progress * facing_mask



def obstacle_avoidance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float):
    """Penalize the robot if ANY ray detects an obstacle closer than the threshold."""
    distances = mdp.get_raycast_distances(env, sensor_cfg)
    
    # Calculate how much the robot has "violated" the threshold
    # 0.0 if safe, >0.0 if inside the threshold
    # violation = torch.clamp(threshold - distances, min=0.0)
    
    # Use the max violation (the single closest point) or mean of all violating rays
    # This creates a "gradient" the robot can follow to move away
    # return torch.max(violation, dim=1)[0] 

    penalty = torch.exp(-distances / (threshold / 2.0))
    
    return torch.mean(penalty, dim=1)

