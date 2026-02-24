from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def rel_target_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Target position in robot's USD local frame."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    pos_w = target.data.root_pos_w - robot.data.root_pos_w
    q_inv = quat_inv(robot.data.root_quat_w)
    return quat_apply(q_inv, pos_w)


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg):
    """Angle to target where 0.0 rad is the -Y axis (The Face)."""
    local_pos = rel_target_pos(env, robot_cfg, target_cfg)
    # Standard atan2(y, x) is for +X. For -Y front, we use:
    # atan2(Side_X, Forward_-Y)
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    return angle.unsqueeze(-1)


def target_reached(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, distance: float) -> torch.Tensor:
    """Checks if the robot is within a specific radius of the target."""
    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    # Calculate Euclidean distance (L2 norm) in world frame
    # root_pos_w is (num_envs, 3)
    tgt_dist = torch.norm(target.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    
    # Return a boolean tensor (Shape: [num_envs])
    # IMPORTANT: Isaac Lab Terminations expect Bool, Rewards cast this to Float automatically
    return tgt_dist <= distance


def lidar_distances(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float = 0.5):
    """Returns normalized Lidar distances [0, 1]. 0 is hit, 1 is clear."""
    raycaster = env.scene[sensor_cfg.name]
    
    # Calculate distance from sensor to hit points
    # raycaster.data.pos_w: (num_envs, num_rays, 3)
    # raycaster.data.sensor_pos_w: (num_envs, 3)
    hit_positions = raycaster.data.ray_hits_w
    sensor_pos = raycaster.data.pos_w.unsqueeze(1)
    
    # Calculate Euclidean distance
    distances = torch.norm(hit_positions - sensor_pos, dim=-1)
    
    # Clip and normalize. 0.0 means touching a wall, 1.0 means clear path (>0.5m)
    return torch.clamp(distances, max=max_distance) / max_distance


def contact_sensor_threshold_filtered(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Terminates ONLY if force comes from the filtered objects (Obstacles).
    Ignores the ground contact from your caster wheel.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    
    # force_matrix_w shape: (num_envs, num_bodies, num_filters, 3)
    # This matrix ONLY populates indices for objects in your 'filter_prim_paths_expr'
    filtered_contact_forces = sensor.data.force_matrix_w
    
    # Calculate the magnitude of forces coming ONLY from the filtered obstacles
    # We take the norm across the XYZ (last dim) and sum across bodies/filters
    force_mag = torch.norm(filtered_contact_forces, dim=-1)
    max_force = torch.max(force_mag.view(env.num_envs, -1), dim=-1)[0]
    
    return max_force > threshold
