from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from warp import quat_rotate_inv
from ict_bot.tasks.obstacle_avoidance.mdp.observations import ray_distances, heading_error
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def progress_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Reward based on moving closer to the target compared to the previous step."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Current distance
    current_dist = torch.norm(target.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    
    # Initialize previous distance if not exists
    if not hasattr(env, "prev_tgt_dist"):
        env.prev_tgt_dist = current_dist.clone()
    
    # 3. Calculate the delta (previous - current)
    # Positive if getting closer, Negative if moving away
    dist_delta = env.prev_tgt_dist - current_dist
    
    # 4. Update the stored distance for the next step
    # We use .clone() to ensure we don't just point to the same tensor
    env.prev_tgt_dist = current_dist.clone()

    # 2. Get Heading Alignment (Cosine of error)
    # 1.0 = Facing target, -1.0 = Facing away
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    alignment = torch.cos(error_rad)

    # 3. ASYMMETRIC LOGIC:
    # If getting closer (dist_delta > 0) AND facing target (alignment > 0.7): reward
    # If getting closer but facing WRONG way: heavy penalty
    # This kills the 'backwards' and 'drifting' strategies immediately.
    gate = (alignment > 0.95).float()
    
    return gate * dist_delta


def reward_turning_priority(env, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    alignment = torch.cos(error_rad)
    
    # Get linear velocity (forward speed)
    lin_vel = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    
    # High reward for alignment ONLY if speed is low
    # This specifically rewards "Stop and Turn"
    return alignment * torch.exp(-lin_vel * 2.0) 


def heading_reward(env, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Reward for facing the target. Returns 1.0 when perfectly aligned, 0.0 at 90 deg."""
    # # 1. Get the local position of the target relative to the robot
    # robot = env.scene[robot_cfg.name]
    # target = env.scene[target_cfg.name]
    
    # pos_rel_w = target.data.root_pos_w - robot.data.root_pos_w
    # local_pos_quat_inv = quat_inv(robot.data.root_quat_w)
    # local_pos = quat_apply(local_pos_quat_inv, pos_rel_w)
    
    # # 2. Calculate the angle in the XY plane
    # angle_to_target = torch.atan2(local_pos[:, 1], local_pos[:, 0])
    
    # # 3. Use Cosine to turn the angle into a [1, -1] reward signal
    # # Perfect alignment (0 rad) = 1.0 reward
    # return torch.cos(angle_to_target)
    """Reward for facing the target. Crucial for steering."""
    # Uses the heading error logic we built (returns radians)
    error = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    # Cosine maps 0 rad to 1.0, and 3.14 rad to -1.0
    return torch.cos(error)


def proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.2):
    """Penalize the robot for being too close to any obstacle detected by Lidar."""
    # Get normalized distances (0.0 to 1.0, where 1.0 = 0.3m)
    norm_distances = ray_distances(env, sensor_cfg)
    
    # Convert normalized back to meters for the threshold check
    # min_dist_m will be between 0.0m and 0.3m
    min_dist_m = torch.min(norm_distances, dim=-1)[0] * 0.3
    
    # Apply penalty if the closest hit is within the threshold (e.g., 0.2m)
    # This gives a linear penalty that scales to -1.0 as the robot touches the object
    penalty = torch.where(
        min_dist_m < threshold, 
        -1.0 * (threshold - min_dist_m) / threshold, 
        torch.zeros_like(min_dist_m)
    )
    
    return penalty


def target_reached(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, distance: float):
    robot = env.scene[asset_cfg.name]
    target = env.scene[target_cfg.name]

    # Calculate Euclidean distance
    tgt_dist = torch.norm(target.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    
    # RETURN TYPE FIX: Must be boolean for self._terminated_buf
    return tgt_dist <= distance 
