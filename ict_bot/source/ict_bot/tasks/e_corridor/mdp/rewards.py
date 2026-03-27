from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def reward_navigate_to_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    h_error = heading_error(env, robot_cfg)
    cos_angle = h_error[:, 1]
    
    # 1. Alignment Factor
    alignment_factor = torch.clamp((cos_angle - 0.966) / (1.0 - 0.966), min=0.0, max=1.0)
    
    # 2. Velocity & Distance Calculation
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)
    target_dir_local = local_pos / (current_dist.unsqueeze(-1) + 1e-6)
    
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    forward_speed = -local_vel[:, 1] # -y is front
    local_vel_2d = local_vel[:, :2]
    target_dir_2d = target_dir_local[:, :2]
    vel_toward_target = (local_vel_2d * target_dir_2d).sum(dim=-1)

    # --- NEW: Velocity Capping (Approach Scaling) ---
    # When distance > 0.5m, scaler is 1.0 (full speed rewarded).
    # When distance < 0.5m, scaler drops linearly to 0.1 at the target.
    approach_scaler = torch.clamp(current_dist / 0.5, min=0.1, max=1.0)

    # --- Aggressive Speed Reward (Now Capped) ---
    # The robot is rewarded for high speed ONLY when far away.
    speed_reward = torch.clamp(forward_speed, min=0.0) * 5.0 * alignment_factor * approach_scaler

    # --- Time/Step Penalty ---
    time_penalty = -0.1 

    # 3. Progress Reward (Now Capped)
    # This prevents the "lunge" by reducing the value of moving fast when close.
    progress = torch.clamp(vel_toward_target, min=0.0) * 20.0 * (0.2 + 0.8 * alignment_factor) * approach_scaler

    # 4. Heading Improvement
    if not hasattr(env, "prev_heading_cos"):
        env.prev_heading_cos = cos_angle.clone()
        heading_imp = torch.zeros(env.num_envs, device=env.device)
    else:
        improvement = cos_angle - env.prev_heading_cos
        if hasattr(env, "reset_buf"):
            improvement = torch.where(env.reset_buf, torch.zeros_like(improvement), improvement)
        env.prev_heading_cos = cos_angle.clone()
        heading_imp = torch.where(cos_angle < 0.966, torch.clamp(improvement, min=0.0) * 40.0, torch.zeros_like(improvement))

    # 5. Backward Penalty
    backward_penalty = torch.where(forward_speed < -0.01, forward_speed * 15.0, torch.zeros_like(forward_speed))

    return progress + speed_reward + heading_imp + backward_penalty + time_penalty


def target_reached_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    reached = check_target_reached(env, robot_cfg)
    terminating_envs = env.reset_buf
    if terminating_envs.any():
        successes = reached[terminating_envs].float().mean()
        env.extras["success_rate"] = 0.98 * env.extras.get(
            "success_rate", torch.tensor(0.0, device=env.device)
        ) + 0.02 * successes
    return reached.float() * 3000.0




def imu_stability_phased(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    # level = getattr(env, "curr_level", 1)
    # if level < 4:
    #     return torch.zeros(env.num_envs, device=env.device)

    imu_data = imu_observations(env, sensor_cfg)
    yaw_rate = torch.abs(imu_data[:, 2])
    
    # Allow a small amount of natural turning (0.5 rad/s) before penalizing
    excess_spin = torch.clamp(yaw_rate - 0.5, min=0.0)
    return -0.5 * excess_spin


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold=0.25):
    level = getattr(env, "curr_level", 1)
    if level < 4:
        return torch.zeros(env.num_envs, device=env.device)
    
    lidar_values = lidar_distances(env, sensor_cfg, max_distance=1.0)
    min_dist, _ = torch.min(lidar_values, dim=-1)
    
    # Soft buffer penalty
    penalty = torch.where(min_dist < (threshold / 0.5), -2.0, 0.0)
    return penalty


def base_posture_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Penalizes the robot lifting its back (pitch deviation)."""
    imu_data = imu_observations(env, sensor_cfg)
    # imu_observations returns [lin_acc x/y/z, ang_vel x/y/z]
    # Pitch rate is ang_vel around X axis (index 3)
    pitch_rate = torch.abs(imu_data[:, 3])
    excess_pitch = torch.clamp(pitch_rate - 0.2, min=0.0)
    return -excess_pitch * 5.0