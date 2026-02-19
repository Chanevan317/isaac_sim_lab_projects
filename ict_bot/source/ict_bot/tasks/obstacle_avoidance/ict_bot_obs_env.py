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
from isaaclab.envs.mdp import root_pos_w, root_quat_w
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
    obstacles = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacles",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "obstacle_avoidance_scene.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Raycaster configuration for obstacle avoidance
    raycaster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacles", 
                is_shared=True, 
                merge_prim_meshes=True, 
                track_mesh_transforms=False
            )
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0), 
            horizontal_fov_range=(0.0, 360.0), 
            horizontal_res=1.2 
        ),
        max_distance=4.0,
        debug_vis=True,
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target_cone",
        spawn=sim_utils.ConeCfg(
            radius=0.2,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0, 0.0, 0.5)),
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
        scale=10.0,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms
        root_pos_w = ObsTerm(func=root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        root_quat_w = ObsTerm(func=root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # Distance to the Target target (Relative position)
        target_relative_pos = ObsTerm(
            func=mdp.get_relative_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "target_cfg": SceneEntityCfg("target")
            }
        )

        # Raycaster Hits (Obstacle distances)
        # This returns the distances measured by your 2x2m grid
        lidar_scan = ObsTerm(
            func=mdp.get_raycast_distances,
            params={"sensor_cfg": SceneEntityCfg("raycaster")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary Goal: Sparse but massive reward for completion
    reaching_target = RewTerm(
        func=mdp.reaching_target_reward,
        weight=50.0,  # Increased from 25.0
        params={"target_cfg": SceneEntityCfg("target")}
    )

    # The "Scent": Dense reward for moving in the right direction
    progress_toward_target = RewTerm(
        func=mdp.progress_reward, 
        weight=15.0,  # Increased from 10.0 to overcome penalties
        params={"target_cfg": SceneEntityCfg("target")}
    )

    # Proximity Penalty: Use a gradient instead of a binary trigger
    # Tip: In your mdp.py, ensure this returns (threshold - distance) 
    # so the penalty grows as the robot gets closer.
    obstacle_penalty = RewTerm(
        func=mdp.obstacle_avoidance_penalty,
        weight=-7.5,
        params={"sensor_cfg": SceneEntityCfg("raycaster"), "threshold": 3.5} # Wider threshold
    )

    # Survival: Encourage speed but keep it low enough to not cause 'suicide' runs
    is_alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.2) 

    # Smoothness: Keep this low so it doesn't prevent movement entirely
    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


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
                # "yaw": (-3.14, 3.14)  # Random heading (Full 360 degrees)
                "yaw": (0.0, 0.0)
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
        params={"asset_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target"), "distance": 0.0}
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