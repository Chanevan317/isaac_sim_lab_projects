from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances, heading_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def reward_velocity_to_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # 1. Target Direction (World)
    pos_w = target.data.root_pos_w - robot.data.root_pos_w
    target_dir_w = pos_w / (torch.norm(pos_w, dim=-1, keepdim=True) + 1e-6)
    
    # 2. Forward Velocity (-Y is your face)
    # We only care about speed in the -Y direction.
    # Positive = Moving Face-First. Negative = Moving Backward.
    forward_speed = -robot.data.root_lin_vel_b[:, 1]
    
    # 3. Alignment (Using your heading_error 0.0 rad = -Y)
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    alignment = torch.cos(error_rad)
    
    # 4. THE FIX: 
    # We ONLY reward forward speed if alignment is positive.
    # If moving backward (forward_speed < 0), this is a penalty.
    # If facing away (alignment < 0), this is a penalty.
    return forward_speed * alignment


def reward_target_clearing(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """
    Rewards having a clear path in the direction of the target.
    Aligned specifically for a -Y front robot.
    """
    # 1. Heading Error (0.0 rad = Target is at -Y)
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    
    # 2. Get Lidar distances (0.0 hit, 1.0 clear)
    norm_dist = lidar_distances(env, sensor_cfg, max_distance=2.0)
    num_rays = norm_dist.shape[1]
    
    # 3. OFFSET LOGIC: 
    # If your -Y face is at the 75% mark (270 degrees) of the Lidar array:
    face_offset_idx = int(num_rays * 0.25)
    
    # Calculate the ray index relative to the -Y face
    # (Heading Error / 2pi) gives the % of the circle to shift
    target_ray_idx = (face_offset_idx + (error_rad / (2 * torch.pi) * num_rays)).long() % num_rays
    
    # 4. Gather the clearance at the ray pointing toward the target
    # We take a small 5-ray average to make it more stable
    indices = torch.stack([(target_ray_idx + i) % num_rays for i in range(-2, 3)], dim=1)
    target_path_clearance = torch.gather(norm_dist, 1, indices).mean(dim=1)
    
    return target_path_clearance


def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """The 'No-Go' Tax: Heavily punishes ANY velocity in the +Y direction."""
    robot = env.scene[robot_cfg.name]
    # Local Y velocity. +Y is backward for your robot.
    backward_vel = robot.data.root_lin_vel_b[:, 1]
    
    # Squared penalty: Reversing at 0.2m/s = -0.04 points, but 1.0m/s = -1.0 points.
    # This makes the robot 'fear' fast reversing more than a slow wobble at spawn.
    return -torch.square(torch.clamp(backward_vel, min=0.0))
