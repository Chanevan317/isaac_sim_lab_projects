from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode terminates if max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    """Terminate if contact force on any link exceeds a threshold."""
    # 1. Access the ContactSensor object
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. Get the current net forces (Shape: [num_envs, num_bodies, 3])
    # Note: We take the norm of the (x,y,z) forces
    net_contact_forces = contact_sensor.data.net_forces_w
    
    # 3. Calculate force magnitudes and check against threshold
    # We check if ANY body on the robot is experiencing force > threshold
    force_magnitudes = torch.norm(net_contact_forces, dim=-1)
    return torch.any(force_magnitudes > threshold, dim=1)


def target_reached(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, distance: float):
    """Terminate if the robot is close enough to the target cone."""
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :2]
    return torch.norm(target_pos - robot_pos, dim=-1) < distance