# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence

from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG
from ict_bot.tasks.move_straight.ict_bot_env import MoveStraightSceneCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
import isaaclab.sim as sim_utils

import os
from ict_bot import ICT_BOT_ASSETS_DIR


# import mdp
import ict_bot.tasks.obstacle_avoidance.mdp as mdp
from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


##
# Scene definition
##


@configclass
class ObstacleAvoidanceSceneCfg(MoveStraightSceneCfg):
    """Configuration for the scene."""
    
    def __post_init__(self):
        super().__post_init__()

    # obstacle avoidance scene assets
    # obstacles = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/obstacles",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "obstacle_avoidance_scene.usd"),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    # )

    # # Raycaster configuration for obstacle avoidance
    # raycaster = MultiMeshRayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base",
    #     offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    #     mesh_prim_paths=[
    #         MultiMeshRayCasterCfg.RaycastTargetCfg(
    #             prim_expr="{ENV_REGEX_NS}/obstacles", 
    #             is_shared=True, 
    #             merge_prim_meshes=True, 
    #             track_mesh_transforms=False
    #         )
    #     ],
    #     pattern_cfg=patterns.LidarPatternCfg(
    #         channels=1, 
    #         vertical_fov_range=(0.0, 0.0), 
    #         horizontal_fov_range=(0.0, 360.0), 
    #         horizontal_res=1.2 
    #     ),
    #     max_distance=4.0,
    #     debug_vis=True,
    # )

    # Target cone configuration
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target_cone",
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, # Keeps it on the floor
                max_depenetration_velocity=1.0,
            ),
            # This allows it to hit the floor but we can ignore it in the robot's logic
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.15)),
    )



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint", "left_wheel_joint"],
        scale=6.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""


    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Lidar/Raycast Data (Obstacle awareness)
        # Pulls the raw distance data from the raycaster sensor defined in the scene
        # lidar_distances = ObsTerm(
        #     func=mdp.ray_distances,
        #     params={"sensor_cfg": SceneEntityCfg("raycaster"), "num_rays": 300, "max_distance": 4.0}
        # )

        # Relative Target Position (Local Frame)
        # Computes the vector from robot to target in the robot's local coordinate system
        rel_pos_to_target = ObsTerm(
            func=mdp.rel_target_pos,
            scale=0.33, # Scale down to keep values in a reasonable range (max ~3m -> 1.0)
            params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}
        )

        # Base Linear and Angular Velocity
        # Tells the robot how fast it is currently moving
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # Heading Error
        # The orientation difference between current heading and target direction
        # heading_to_target = ObsTerm(
        #     func=mdp.heading_error,
        #     params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}
        # )

        # Previous Action
        # Useful for smoothing and understanding current momentum
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Progress Reward (Potential)
    # Scaled high to encourage movement
    progress = RewTerm(
        func=mdp.progress_reward,
        weight=3000.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}
    )

    face_target = RewTerm(
        func=mdp.reward_turning_priority,
        weight=20.0, 
        params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")}
    )

    # Heading Reward
    # Encourages facing the target. Uses the error function we defined for observations.
    # heading_alignment = RewTerm(
    #     func=mdp.heading_reward,
    #     weight=5.0,
    #     params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")},
    # )

    # Target Reach Bonus (Sparse)
    # Triggered by the same logic as the success termination
    target_reach_bonus = RewTerm(
        func=mdp.target_reached,
        weight=100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"), 
            "target_cfg": SceneEntityCfg("target"), 
            "distance": 0.2
        }
    )

    # Proximity Penalty (Soft buffer)
    # Encourages staying at least 0.25m away from obstacles
    # obstacle_proximity = RewTerm(
    #     func=mdp.proximity_penalty,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("raycaster"), "threshold": 0.25}
    # )

    # Action Smoothing
    # Penalizes sudden changes in wheel velocities (jitter)
    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.01
    # )


    # alive_penalty = RewTerm(
    #     func=mdp.is_alive,
    #     weight=-0.1
    # )


@configclass
class MyEventCfg:
    
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0), 
                "y": (0.0, 0.0), 
                "z": (0.2, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),  # Random heading (Full 360 degrees)
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {}, # Sets all velocities to 0
        },
    )

    reset_target_position = EventTerm(
        func=mdp.reset_target_in_ring,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "radius_range": (2.5, 2.6), # 2.5m to 2.6m distance
            "z_height": 0.6            # Exactly half the cone height
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

    # Terminate if the robot reaches the target (Success Reset)
    # This speeds up training so the robot learns to find a *new* target immediately
    target_reached = DoneTerm(
        func=mdp.target_reached,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target"), "distance": 0.2}
    )


##
# Environment configuration
##


@configclass
class ObstacleAvoidanceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot to move straight."""

    # Scene settings
    scene: ObstacleAvoidanceSceneCfg = ObstacleAvoidanceSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: MyEventCfg = MyEventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0


##
# Environment class
##


class ObstacleAvoidanceEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: ObstacleAvoidanceEnvCfg
    
    def __init__(self, cfg: ObstacleAvoidanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find wheel joint indices
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
    
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset selected environments."""
        super()._reset_idx(env_ids)
        
        # Handle None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_resets = len(env_ids)
        
        # Reset wheel joint positions and velocities to zero
        num_wheels = len(self._wheel_indices)
        joint_pos = torch.zeros((num_resets, num_wheels), device=self.device)
        joint_vel = torch.zeros((num_resets, num_wheels), device=self.device)
        
        self.scene["robot"].write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._wheel_indices,
            env_ids=env_ids,
        )