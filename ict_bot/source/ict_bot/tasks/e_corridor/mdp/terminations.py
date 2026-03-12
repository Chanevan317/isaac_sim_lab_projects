from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminates episodes at 30s for Ph1-2, and 45s for Ph3+."""
    # Convert current step buffer to seconds
    # (episode_length_buf counts steps, so we multiply by dt * decimation)
    current_time_s = env.episode_length_buf * env.step_dt
    
    # Determine the limit based on the level
    # Default to 30s if level is 1 or 2
    limit = 30.0
    if getattr(env, "curr_level", 1) >= 3:
        limit = 45.0
        
    # Return a boolean tensor of environments that exceeded their specific limit
    return current_time_s >= limit