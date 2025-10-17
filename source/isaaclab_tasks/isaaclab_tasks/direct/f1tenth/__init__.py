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
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",  # RL-Games SAC
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",  # SKRL SAC
    },
)
