from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import RigidObject


def get_raycast_distances(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculates the scalar distance for each ray in the raycaster."""
    # 1. Access the sensor instance from the scene
    # Note: Access via env.scene.sensors if it's a sensor group
    sensor: RayCaster = env.scene[sensor_cfg.name]
    
    # 2. Get the hit positions in World Frame [num_envs, num_rays, 3]
    hit_pos_w = sensor.data.ray_hits_w
    
    # 3. Get the sensor's own position in World Frame [num_envs, 3]
    sensor_pos_w = sensor.data.pos_w
    
    # 4. Calculate Euclidean distance: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    # We unsqueeze sensor_pos_w to [num_envs, 1, 3] to broadcast across all rays
    distances = torch.norm(hit_pos_w - sensor_pos_w.unsqueeze(1), dim=-1)
    
    # 5. Optional: Clip the distance to the max range defined in your config
    # This prevents 'infinity' values if a ray hits nothing
    max_range = sensor.cfg.max_distance
    return torch.clamp(distances, max=max_range)


def get_relative_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """Calculates the position of the target relative to the robot's body frame."""
    # Get world positions
    robot: RigidObject = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    # Position of target relative to robot in World Frame
    relative_pos_w = target.data.root_pos_w - robot.data.root_pos_w
    
    # Rotate the relative position into the Robot's Local Frame
    # (Uses the inverse of the robot's world orientation)
    robot_quat_w = robot.data.root_quat_w
    relative_pos_b = math_utils.quat_apply_inverse(robot_quat_w, relative_pos_w)
    
    return relative_pos_b