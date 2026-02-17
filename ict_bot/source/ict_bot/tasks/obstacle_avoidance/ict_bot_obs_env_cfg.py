# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG
from ict_bot.tasks.obstacle_avoidance.ict_bot_obs_env import ObstacleAvoidanceEnvCfg

##
# Scene definition
##


@configclass
class IctBotObsEnvCfg(ObstacleAvoidanceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 457  # [Base(13) + Target_Rel_Pos(3) + Raycaster(441)] = 457
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.16
    max_linear_velocity = 0.5
    max_angular_velocity = 2.0
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    
    # Reset conditions
    yaw_limit = 3.14  # Allowed to turn fully (180 degrees) to find the cone


@configclass
class IctBotObsEnvCfg_PLAY(IctBotObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False