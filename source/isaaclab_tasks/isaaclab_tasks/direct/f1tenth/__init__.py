# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
F1TENTH racing environment with LiDAR-based navigation.
"""

import gymnasium as gym

from . import agents
from .f1tenth_env import F1TenthEnv, F1TenthEnvCfg

##
# Register Gym environments
##

gym.register(
    id="Isaac-F1tenth-Direct-v0",
    entry_point=f"{__name__}.f1tenth_env:F1TenthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.f1tenth_env:F1TenthEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",  # Default (for PPO fallback)
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",  # SAC specific
    },
)
