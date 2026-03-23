from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.e_corridor.mdp.observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from isaaclab.utils.math import quat_inv, quat_apply
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reward_gated_progress_exponential(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)
    dist_delta = (env.prev_tgt_dist - current_dist) / env.step_dt
    
    if level == 1:
        return torch.clamp(dist_delta, min=0.0) * 15.0 

    # --- DYNAMIC GATING ---
    if level == 2:
        pow_val = 2.0
    elif level in [3, 4]:
        # WIDE GATE: Pow 1.0 creates a linear, gentle slope for discovery.
        pow_val = 1.0 
    else: 
        # Level 5/6: TIGHT GATE (Pow 4.0) for corridor precision.
        pow_val = 4.0
    
    h_error = heading_error(env, robot_cfg)
    alignment_gate = torch.pow(torch.clamp(h_error[:, 1], min=0.0), pow_val)
    
    # SPEED GATE: Kept at 0.05 for all levels. 
    # If you raise this in Phase 6, the robot will be penalized for driving 
    # carefully around tight corners.
    speed_gate = torch.sigmoid(10.0 * (dist_delta - 0.05))
    
    return torch.clamp(dist_delta, min=0.0) * alignment_gate * speed_gate * 20.0


def forward_velocity_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    robot = env.scene[robot_cfg.name]
    
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    forward_speed = -local_vel[:, 1] # Local -Y is front
    
    # UNIVERSAL BACKWARD PENALTY
    # This completely destroys the "drive backward to goal" cheat in Phase 4-6.
    backward_penalty = torch.where(forward_speed < 0, forward_speed * 15.0, 0.0)
    positive_speed = torch.clamp(forward_speed, min=0.0)
    
    h_error = heading_error(env, robot_cfg)
    alignment = torch.clamp(h_error[:, 1], min=0.0)
    
    if level == 1:
        reward = positive_speed * 15.0
    elif level == 2:
        reward = positive_speed * alignment * 10.0
    elif level == 3:
        gate = torch.pow(alignment, 1.5) 
        reward = positive_speed * gate * 15.0 
    elif level == 4:
        # PIVOT PHASE: Maximize reward weight to pull it through the 180 turn
        gate = torch.pow(alignment, 1.0) 
        reward = positive_speed * gate * 25.0 
    else: # Phase 5/6
        # MASTERY: High weight so it isn't afraid to drive, but strict gate.
        gate = torch.pow(alignment, 2.0)
        reward = positive_speed * gate * 20.0
        
    return reward + backward_penalty


def target_reached_reward_phased(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, distance=0.3):
    level = getattr(env, "curr_level", 1)
    robot = env.scene[robot_cfg.name]
    dist = torch.norm(env.target_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    
    reached = (dist < distance).float()

    # Smoothed out the scaling. A jump from 10k to 25k can destabilize the value network.
    prizes = {1: 500.0, 2: 1000.0, 3: 3000.0, 4: 6000.0, 5: 10000.0, 6: 15000.0}
    prize = prizes.get(level, 15000.0)
        
    return reached * prize


def imu_stability_phased(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    if level < 5:
        return torch.zeros(env.num_envs, device=env.device)

    imu_data = imu_observations(env, sensor_cfg)
    yaw_rate = torch.abs(imu_data[:, 2])
    
    # Allow a small amount of natural turning (0.5 rad/s) before penalizing
    excess_spin = torch.clamp(yaw_rate - 0.5, min=0.0)
    return -0.5 * excess_spin


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold=0.25):
    level = getattr(env, "curr_level", 1)
    if level < 6:
        return torch.zeros(env.num_envs, device=env.device)
    
    lidar_values = lidar_distances(env, sensor_cfg, max_distance=1.0)
    min_dist, _ = torch.min(lidar_values, dim=-1)
    
    # Soft buffer penalty
    penalty = torch.where(min_dist < (threshold / 0.5), -2.0, 0.0)
    return penalty


def action_rate_l2_phased(env: ManagerBasedRLEnv):
    level = getattr(env, "curr_level", 1)
    if level == 1:
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = action_rate_l2(env)
    weight = 0.01 if level <= 4 else 0.05
    return penalty * weight


def is_alive_phased(env: ManagerBasedRLEnv):
    """
    Softened to prevent the agent from intentionally crashing to escape 
    the heavy time penalties during difficult pivot learning.
    """
    level = getattr(env, "curr_level", 1)
    
    if level == 1: weight = -0.5
    elif level == 2: weight = -1.0
    elif level in [3, 4]: weight = -2.0 # Give it time to figure out the turn
    else: weight = -3.0 # Increase urgency in the corridor
        
    return torch.ones(env.num_envs, device=env.device) * weight


def joint_vel_penalty_phased(env: ManagerBasedRLEnv):
    level = getattr(env, "curr_level", 1)
    if level == 1:
        return torch.zeros(env.num_envs, device=env.device)
    
    raw_penalty = joint_vel_l2(env)
    weight = -0.0001 if level == 2 else -0.0005
    return raw_penalty * weight


def base_posture_penalty(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_height=0.1):
    """
    NEW: Stops the robot from "lifting its back" or balancing upwards.
    Adjust `target_height` to match your robot's normal resting Z-coordinate.
    """
    level = getattr(env, "curr_level", 1)
    if level < 3:
        return torch.zeros(env.num_envs, device=env.device)
        
    robot = env.scene[robot_cfg.name]
    base_z = robot.data.root_pos_w[:, 2]
    
    # Penalize deviation from normal driving height
    height_error = torch.square(base_z - target_height)
    return -height_error * 50.0 # Heavy penalty for physical exploits