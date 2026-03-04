from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def reset_camera_posture_uniform(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    sensor_cfg: SceneEntityCfg, 
    z_range: tuple[float, float], 
    pitch_range: tuple[float, float]
):
    """Randomizes the camera's height and tilt (pitch) relative to its parent link."""
    # 1. Extract the sensor object
    tiled_camera = env.scene[sensor_cfg.name]
    num_resets = len(env_ids)

    # 2. Generate random values
    rand_z = (z_range[1] - z_range[0]) * torch.rand(num_resets, device=env.device) + z_range[0]
    rand_pitch = (pitch_range[1] - pitch_range[0]) * torch.rand(num_resets, device=env.device) + pitch_range[0]

    # 3. Apply Height (Z-axis)
    # We modify the world position's Z component for the resetting envs
    tiled_camera.data.pos_w[env_ids, 2] += rand_z

    # 4. Apply Pitch (Tilt)
    # Create a quaternion for the random pitch (rotation around Y)
    # Order is (roll, pitch, yaw)
    pitch_quat = quat_from_euler_xyz(
        torch.zeros(num_resets, device=env.device),
        rand_pitch,
        torch.zeros(num_resets, device=env.device)
    )

    # Multiply the current world orientation by the new random pitch
    # quat_mul handles the vectorized multiplication for all env_ids
    tiled_camera.data.quat_w[env_ids] = quat_mul(tiled_camera.data.quat_w[env_ids], pitch_quat)


def update_camera_fov_uniform(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    sensor_cfg: SceneEntityCfg, 
    fov_range: tuple[float, float]
):
    """Randomizes the Horizontal FOV by modifying the camera's focal length."""
    tiled_camera = env.scene[sensor_cfg.name]
    num_resets = len(env_ids)
    
    # 1. Generate new FOV values (in degrees)
    new_fov_deg = (fov_range[1] - fov_range[0]) * torch.rand(num_resets, device=env.device) + fov_range[0]
    
    # 2. Convert FOV to Focal Length (Simplified pinhole model)
    # Math: focal_length = (aperture / 2) / tan(fov / 2)
    # In Isaac Lab, we often modify the horizontal_aperture or focal_length directly.
    new_focal_lengths = 24.0 * (60.0 / new_fov_deg) # Ratio-based adjustment for C270 baseline
    
    # 3. Apply the change to the sensor intrinsics
    # This requires the sensor to re-calculate its projection matrix
    tiled_camera.set_intrinsic_matrices(focal_lengths=new_focal_lengths, env_ids=env_ids)