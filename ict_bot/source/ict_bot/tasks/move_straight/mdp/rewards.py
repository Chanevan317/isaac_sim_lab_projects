from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def forward_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward POSITIVE X-axis velocity (world frame)."""
    vel_x = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return torch.clamp(vel_x, min=0)


def backward_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize NEGATIVE X-axis velocity (world frame)."""
    vel_x = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return torch.clamp(-vel_x, min=0)


def straightness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize Y-axis velocity (lateral motion in world frame)."""
    vel_y = env.scene["robot"].data.root_lin_vel_w[:, 1]
    return torch.abs(vel_y)


def heading_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize excessive turning/spinning (Z angular velocity)."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_w[:, 2]
    return torch.abs(yaw_rate)


def idle_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize if robot is not moving forward (low X velocity)."""
    vel_x = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return (torch.abs(vel_x) < 0.01).float()