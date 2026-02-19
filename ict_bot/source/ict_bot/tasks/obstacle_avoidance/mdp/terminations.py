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


def target_reached(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, distance: float):
    """Terminate if the robot is close enough to the target cone."""
    robot_pos = env.scene[asset_cfg.name].data.root_pos_w[:, :2]
    target_pos = env.scene[target_cfg.name].data.root_pos_w[:, :2]
    return torch.norm(target_pos - robot_pos, dim=-1) < distance