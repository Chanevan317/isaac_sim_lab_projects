from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances, heading_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reward_robust_navigation(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """
    Robust Setup for -Y Face (270-degree Lidar Offset):
    1. Forward Speed (-Y)
    2. Alignment (-Y to Target)
    3. Target-Ray Clearance (Gap Seeking via 270 deg offset)
    4. Symmetry (Centering via 270 deg offset)
    """
    robot = env.scene[robot_cfg.name]
    
    # 1. Forward Speed: -Y is forward. Positive = moving 'Face-First'.
    # Clamping at 0.0 prevents the 'Double Negative' backward loophole.
    forward_speed = torch.clamp(-robot.data.root_lin_vel_b[:, 1], min=0.0)
    
    # 2. Alignment: Using your heading_error (0.0 rad = -Y face pointing at target)
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    alignment = torch.clamp(torch.cos(error_rad), min=0.0)
    
    # 3. Gap Seeking: Get clearance specifically in the direction of the target
    # We use max_distance=2.0 for foresight through the 1m gaps
    norm_dist = lidar_distances(env, sensor_cfg, max_distance=2.0)
    num_rays = norm_dist.shape[1]
    
    # --- COORDINATE SYNC: -Y is at 270 degrees (0.75 index) ---
    face_idx = int(num_rays * 0.75) 
    
    # Find the ray pointing to the goal, relative to the -Y face
    target_ray_idx = (face_idx + (error_rad / (2 * torch.pi) * num_rays)).long() % num_rays
    target_path_clearance = torch.gather(norm_dist, 1, target_ray_idx.unsqueeze(-1)).squeeze(-1)
    
    # 4. Lane Centering: Reward symmetry between left and right front rays
    # This pushes the robot to the middle of the 1m gaps.
    left_ray = norm_dist[:, (face_idx - 15) % num_rays]
    right_ray = norm_dist[:, (face_idx + 15) % num_rays]
    centering_reward = 1.0 - torch.abs(left_ray - right_ray)

    # --- THE MULTIPLIER ---
    # Total Reward = (Speed * Alignment * PathToTargetClearance)
    drive_reward = forward_speed * alignment * target_path_clearance
    
    # Scale: Speed (0.6) * Align (1.0) * Clear (1.0) = 0.6 max per step
    return (drive_reward * 100.0) + (centering_reward * 2.0)


def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Heavy 'No-Go' Tax on +Y velocity to kill the backward habit."""
    robot = env.scene[robot_cfg.name]
    backward_vel = robot.data.root_lin_vel_b[:, 1] # +Y is backward
    return -torch.square(torch.clamp(backward_vel, min=0.0))
