from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def rel_target_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Vector from robot to target in world frame
    pos_rel_w = target.data.root_pos_w - robot.data.root_pos_w
    
    # Invert the robot's orientation to get the "local" transform
    # quat_inv is standard in 2.3.2
    robot_quat_inv = quat_inv(robot.data.root_quat_w)
    
    # Rotate the relative vector by the inverse quaternion
    return quat_apply(robot_quat_inv, pos_rel_w)


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    # Get the local relative position
    local_pos = rel_target_pos(env, robot_cfg, target_cfg)
    # Angle to target in local XY plane
    return torch.atan2(local_pos[:, 1], local_pos[:, 0]).unsqueeze(-1)


def ray_distances(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, num_rays: int = 300, max_distance: float = 4.0):
    """Returns the ray distances clipped and normalized by a specific range."""
    # Access the MultiMeshRayCaster sensor
    raycaster = env.scene[sensor_cfg.name]
    
    # 1. Get Hit Positions in World Frame (N, B, 3)
    # If your version uses 'ray_hits_w', use that. 
    # Most Isaac Lab versions store it in 'pos_w'
    hit_positions = raycaster.data.ray_hits_w
    
    # 2. Get Sensor Position in World Frame (N, 3)
    sensor_pos = raycaster.data.pos_w.unsqueeze(1) # Shape (N, 1, 3)
    
    # 3. Calculate Euclidean Distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    # Shape: (num_envs, num_rays)
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    
    # 4. Handle Ray Slicing/Padding
    curr_rays = distances.shape[1]
    if curr_rays > num_rays:
        distances = distances[:, :num_rays]
    elif curr_rays < num_rays:
        padding = torch.full((distances.shape[0], num_rays - curr_rays), max_distance, device=env.device)
        distances = torch.cat([distances, padding], dim=1)
    
    # 5. Clip and Normalize (0.0 to 1.0)
    clamped_distances = torch.clamp(distances, max=max_distance) / max_distance
    
    return clamped_distances