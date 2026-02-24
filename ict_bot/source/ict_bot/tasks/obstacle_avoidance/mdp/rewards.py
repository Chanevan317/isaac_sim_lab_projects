from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.obstacle_avoidance.mdp.observations import rel_target_pos, lidar_distances, heading_error
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



# def heading_error_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
#     """Reward: High when -Y face is aligned with target. Returns flat [N] tensor."""
#     # 1. Call the observation function: returns [4096, 1]
#     error_obs = heading_error(env, robot_cfg, target_cfg)
    
#     # 2. Squeeze the last dimension: returns [4096]
#     error_rad = error_obs.squeeze(-1)
    
#     # 3. Apply Cosine: 1.0 (facing) to -1.0 (away)
#     return torch.cos(error_rad)


# def reward_gated_progress_neg_y(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
#     """Rewards moving closer ONLY if the -Y face is pointed at the target."""
#     # 1. Distance Delta
#     current_dist = torch.norm(env.scene[target_cfg.name].data.root_pos_w - env.scene[robot_cfg.name].data.root_pos_w, dim=-1)
#     if not hasattr(env, "prev_tgt_dist"):
#         env.prev_tgt_dist = current_dist.clone()
#     dist_delta = env.prev_tgt_dist - current_dist
#     env.prev_tgt_dist = current_dist.clone()

#     # 2. Strict Alignment (Must be within ~11 degrees)
#     error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
#     alignment = torch.cos(error_rad)
    
#     # 3. THE GATE: 0.98 is a very tight window.
#     # If the robot isn't pointing its 'face' almost exactly at the cone, 
#     # it gets ZERO progress reward.
#     # gate = (alignment > 0.98).float()
    
#     # Also, we penalize moving closer if NOT aligned (The 'Anti-Spiral' rule)
#     # return torch.where(gate > 0, dist_delta * 5.0, -torch.abs(dist_delta) * 2.0)

#     return torch.where(alignment > 0.0, dist_delta * alignment, -torch.abs(dist_delta))


def reward_directional_progress(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Behavior 1: Projected Velocity Gate."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Vector to target (World)
    pos_w = target.data.root_pos_w - robot.data.root_pos_w
    target_dir = pos_w / (torch.norm(pos_w, dim=-1, keepdim=True) + 1e-6)
    
    # World Velocity
    world_vel = robot.data.root_lin_vel_w
    
    # Projection: Moving toward goal?
    proj_vel = torch.sum(world_vel * target_dir, dim=-1)
    
    # Alignment: Facing goal with -Y? 
    # (1.0 = Facing, -1.0 = Opposite)
    error_rad = heading_error(env, robot_cfg, target_cfg).squeeze(-1)
    alignment = torch.cos(error_rad)
    
    # This result is POSITIVE only if (moving toward goal AND facing it).
    # If moving toward goal but BACKING UP, alignment is negative, so reward is negative.
    return proj_vel * alignment

def reward_virtual_bumper(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Behavior 2: Linear-Inverse Repulsion Field."""
    # Lidar distance [0, 1] where 1 is clear (max_dist=1.5m)
    norm_dist = lidar_distances(env, sensor_cfg, max_distance=1.5)
    min_dist, _ = torch.min(norm_dist, dim=-1)
    
    # Penalty ramps up as robot gets closer than 0.75m (0.5 normalized)
    # Result: 0.0 (safe) to -1.0 (collision)
    return torch.clamp((min_dist - 0.5) * 2.0, max=0.0)

def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Behavior 3: Anti-Reverse Kill Switch."""
    robot = env.scene[robot_cfg.name]
    # Local Y velocity. +Y is backward.
    backward_vel = robot.data.root_lin_vel_b[:, 1]
    # Squared penalty for any backward motion
    return -torch.square(torch.clamp(backward_vel, min=0.0))

def reward_time_tax(env: ManagerBasedRLEnv):
    """Behavior 4: The Gas Pedal (Constant Pressure)."""
    return -torch.ones(env.num_envs, device=env.device)

