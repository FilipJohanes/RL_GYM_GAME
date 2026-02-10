from __future__ import annotations

import gymnasium as gym


ENV_ID = "FrozenLake-v1"
IS_SLIPPERY = False


def make_env(render_mode=None) -> gym.Env:
    return gym.make(ENV_ID, is_slippery=IS_SLIPPERY, render_mode=render_mode)
