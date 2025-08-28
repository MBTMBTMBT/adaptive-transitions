# experiment_utils/_env_factories.py
# English comments only.
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from apis.customisable import CustomisableEnvAbs
from customised_minigrid_env import CustomMiniGridEnv
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from networkx_env.networkx_env import NetworkXMDPEnvironment


class IntegerStateObsWrapper(gym.ObservationWrapper):
    """
    Replace observations with a single integer code produced by env.encode_state().

    - Observation space is Box(low=0, high=int64_max, shape=(1,), dtype=int64).
    - Optionally, keep the original observation in info["raw_obs"] for debugging.
    - Works well with Stable-Baselines3 DummyVecEnv, avoiding Text/Dict obs issues.
    """

    def __init__(
        self,
        env: CustomisableEnvAbs,
        keep_raw_obs_in_info: bool = False,
        use_int32: bool = False,
    ):
        super().__init__(env)
        self.keep_raw_obs_in_info = bool(keep_raw_obs_in_info)
        self._dtype = np.int32 if use_int32 else np.int64

        # Default upper bound = dtype max
        n = int(np.iinfo(self._dtype).max)

        # Discrete(n) -> valid observations in {0, ..., n-1}
        self.observation_space = spaces.Discrete(n, start=0)

        # get the actual MDP network
        self.mdp_network = None
        if hasattr(self.env, "get_mdp_network"):
            self.mdp_network = self.env.get_mdp_network()
            self.observation_space = spaces.Discrete(len(self.mdp_network.states))

    def _encode_to_array(self) -> np.ndarray:
        code = int(
            self.env.encode_state()
        )  # rely on your CustomMiniGridEnv.encode_state()
        return np.asarray([code], dtype=self._dtype)

    # ---- ObservationWrapper hooks ----
    def observation(self, observation):
        # Note: we ignore 'observation' from inner env and return the integer code instead.
        return self._encode_to_array()

    # ---- Override reset/step so we can optionally stash raw obs in info ----
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw_obs, info = self.env.reset(**kwargs)
        obs = self._encode_to_array()
        if self.keep_raw_obs_in_info:
            info = dict(info or {})
            info["raw_obs"] = raw_obs
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._encode_to_array()
        if self.keep_raw_obs_in_info:
            info = dict(info or {})
            info["raw_obs"] = raw_obs
        return obs, float(reward), bool(terminated), bool(truncated), info


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
        map_name=map_name,
        is_slippery=is_slippery,
        networkx_env=None,
        render_mode="rgb_array",
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
        is_rainy=False,
        fickle_passenger=False,
        networkx_env=None,
        render_mode="rgb_array",
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
        display_mode="middle",
        any_key_opens_the_door=False,
        networkx_env=None,
        render_mode="rgb_array",
    )
    env = IntegerStateObsWrapper(env, keep_raw_obs_in_info=False)
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
