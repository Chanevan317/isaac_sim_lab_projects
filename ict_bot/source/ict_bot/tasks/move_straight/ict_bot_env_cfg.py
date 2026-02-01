# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class IctBotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    # - spaces definition
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 12  # Matches the 'obs' tensor we build
    state_space = 0          # Usually 0 unless using asymmetric actor-critic

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = ICT_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # robot_cfg: ArticulationCfg = ICT_BOT_CFG.replace(prim_path="/World/ict_bot")

    # scene
    num_envs=1024
    env_spacing=5.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs, 
        env_spacing=env_spacing, 
        replicate_physics=True
    )
    
    # Physical properties
    wheel_radius = 1.0
    wheel_spacing = 1.6
    max_linear_velocity = 0.5   # meters per second
    max_angular_velocity = 2.0  # radians per second

    # custom parameters/scales
    # - controllable joint
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    # - action scale
    action_scale = 15.0  # [N]
    # - reward scales
    reward_scales = {
        "progress_reward" : 1.0,        # Reward for velocity along the X-axis 
        "straightness_penalty" : -0.5,  # Penalty for velocity along the Y-axis (drifting)
        "heading_penalty" : -0.2,       # Penalty for angular velocity (turning)
        "termination": -10.0            # Penalty for hitting reset limits
    }
    # - reset states/conditions
    y_drift_limit = 1.0       # Reset if robot drifts > 1m from center
    yaw_limit = 0.785         # Reset if robot turns > 45 degrees (pi/4)


@configclass
class IctBotEnvCfg_Play(DirectRLEnvCfg):
    num_envs = 4
    env_spacing = 5
    episode_length_s = 60.0