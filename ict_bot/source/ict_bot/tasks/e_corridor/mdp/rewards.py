from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.e_corridor.mdp.observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def progress_to_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Rewards moving closer to the target using the local relative position."""
    # Reuse your existing observation function logic
    local_pos = rel_target_pos(env, robot_cfg)
    
    # Current distance is just the norm of the relative vector
    current_dist = torch.norm(local_pos, dim=-1)
    
    # We want to reward the REDUCTION in distance
    # Isaac Lab typically tracks previous states for you
    distance_moved = env.prev_tgt_dist - current_dist
    return distance_moved / env.step_dt


def align_to_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Rewards facing the target (-Y front)."""
    # Use your heading_error function which returns [sin, cos]
    # cos(angle) is 1.0 when perfectly aligned
    h_error = heading_error(env, robot_cfg)
    cos_theta = h_error[:, 1] 
    return cos_theta


def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Penalize moving in the local +Y direction."""
    robot = env.scene[robot_cfg.name]
    # Local velocity (num_envs, 3)
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    # If Y > 0, it's moving backward. 
    return torch.clamp(local_vel[:, 1], min=0.0)


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.3):
    """Penalizes getting too close to walls based on LIDAR."""
    # Use your normalized lidar_distances [0, 1]
    # 0.0 is a hit, 1.0 is clear (at max_distance=0.5m)
    lidar_values = lidar_distances(env, sensor_cfg, max_distance=0.5)
    
    # Find the closest point in the entire scan
    min_dist, _ = torch.min(lidar_values, dim=-1)
    
    # If min_dist < threshold (e.g., 0.3/0.5 = 0.6 normalized), apply penalty
    # This creates a "soft" buffer around the robot
    penalty = torch.where(min_dist < (threshold / 0.5), -1.0, 0.0)
    return penalty


def imu_stability_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Penalizes high angular velocity to prevent shaking/wobbling."""
    imu_data = imu_observations(env, sensor_cfg)
    
    # imu_data[:, 2] is the Yaw Rate (Z-axis angular velocity)
    # We penalize the absolute value of the yaw rate
    yaw_rate = torch.abs(imu_data[:, 2])
    
    # Small penalty to encourage smooth, straight driving
    return -0.01 * yaw_rate