# experiment_utils/env_factories.py
# English comments only.

from gymnasium.wrappers import TimeLimit

from customised_minigrid_env import CustomMiniGridEnv
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from networkx_env.networkx_env import NetworkXMDPEnvironment

def make_frozenlake_target(seed: int, **kwargs):
    """
    Create the native FrozenLake env wrapped with TimeLimit.
    kwargs:
      - map_name: str (default "8x8")
      - is_slippery: bool (default True)
      - max_steps: int (default 500)
    """
    map_name = kwargs.get("map_name", "8x8")
    is_slippery = bool(kwargs.get("is_slippery", True))
    max_steps = int(kwargs.get("max_steps", 500))

    env = CustomisedFrozenLakeEnv(
        render_mode=None,
        map_name=map_name,
        is_slippery=is_slippery,
        networkx_env=None,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_taxi_target(seed: int, **kwargs):
    """
    Create the native Taxi env wrapped with TimeLimit.
    No rainy/fickle toggles are used here (always False).
    kwargs:
      - max_steps: int
    """
    max_steps = int(kwargs.get("max_steps", 250))

    env = CustomisedTaxiEnv(
        render_mode=None,
        is_rainy=False,
        fickle_passenger=False,
        networkx_env=None,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_minigrid_target(seed: int, **kwargs):
    """
    Create the native MiniGrid env wrapped with TimeLimit.
    kwargs:
      - map_name: str (default "door-key")
      - max_steps: int (default 1000)
    """
    map_name = kwargs.get("map_name", "door-key")
    max_steps = int(kwargs.get("max_steps", 1000))

    env = CustomMiniGridEnv(
        map_name=map_name,
        random_rotate=False,
        random_flip=False,
        any_key_opens_the_door=False,
        networkx_env=None,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_nx_env_from_mdp(mdp, seed: int, **kwargs):
    """
    Create a NetworkX-backed env from a given MDPNetwork, wrapped with TimeLimit.
    kwargs:
      - max_steps: int (default 500)
    """
    max_steps = int(kwargs.get("max_steps", 500))
    env = NetworkXMDPEnvironment(mdp_network=mdp, render_mode=None, seed=seed)
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env
