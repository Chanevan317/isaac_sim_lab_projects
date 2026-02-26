from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reset_target_in_ring(env: ManagerBasedRLEnv, env_ids, asset_cfg: SceneEntityCfg, radius_range: float | tuple, z_height: float):
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    
    # Handle your float/tensor/tuple radius
    if isinstance(radius_range, (float, int, torch.Tensor)):
        r_min, r_max = float(radius_range), float(radius_range)
    else:
        r_min, r_max = radius_range
    
    # Sample Angle and Radius
    theta = 2.0 * torch.pi * torch.rand(num_resets, device=env.device)
    if r_min == r_max:
        rand_r = r_min
    else:
        rand_r = torch.sqrt(torch.rand(num_resets, device=env.device) * (r_max**2 - r_min**2) + r_min**2)
    
    # Calculate LOCAL x, y
    local_x = rand_r * torch.cos(theta)
    local_y = rand_r * torch.sin(theta)
    
    # GET THE ORIGINS for the specific env_ids
    # This is the crucial missing step!
    env_origins = env.scene.env_origins[env_ids]
    
    # Add Local Offset to the Global Origin
    target_pos_w = env_origins.clone()
    target_pos_w[:, 0] += local_x
    target_pos_w[:, 1] += local_y
    target_pos_w[:, 2] = z_height # Usually Z is the same across all envs
    
    # Prepare the root state (pos 3 + quat 4 + lin_vel 3 + ang_vel 3 = 13)
    # We use default_root_state to keep the velocities at zero
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = target_pos_w
    
    # Force upright rotation (Identity Quat)
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # Write to simulation
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)