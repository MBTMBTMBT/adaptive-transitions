from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from apis.customisable import CustomisableEnvAbs
from customised_minigrid_env import CustomMiniGridEnv
from customised_toy_text_envs.customised_frozenlake import CustomisedFrozenLakeEnv
from customised_toy_text_envs.customised_taxi import CustomisedTaxiEnv
from experiment_utils.utils import _import
from networkx_env.networkx_env import NetworkXMDPEnvironment
from mdp_network.mdp_network import MDPNetwork


class IntegerStateObsWrapper(gym.ObservationWrapper):
    """Turn obs into a single integer via env.encode_state()."""

    def __init__(
        self,
        env: CustomisableEnvAbs,
        keep_raw_obs_in_info: bool = False,
        use_int32: bool = False,
    ):
        super().__init__(env)
        self.keep_raw_obs_in_info = bool(keep_raw_obs_in_info)
        self._dtype = np.int32 if use_int32 else np.int64
        n = int(np.iinfo(self._dtype).max)
        self.observation_space = spaces.Discrete(n, start=0)

        self.mdp_network = None
        if hasattr(self.env, "get_mdp_network"):
            self.mdp_network = self.env.get_mdp_network()
            self.observation_space = spaces.Discrete(len(self.mdp_network.states))

    def _encode_to_array(self) -> np.ndarray:
        code = int(self.env.encode_state())
        return np.asarray([code], dtype=self._dtype)

    def observation(self, observation):
        return self._encode_to_array()

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


def _resolve_networkx_env_or_none(
    seed: Optional[int], cfg: Dict[str, Any]
) -> Optional[NetworkXMDPEnvironment]:
    """
    Resolve an optional NetworkXMDPEnvironment from cfg.
    Accepts either:
      - 'networkx_env': an existing NetworkXMDPEnvironment (takes precedence; exclusive), OR
      - one of the MDP sources (exclusive):
          * 'mdp_config_path': str
          * 'mdp_config_data': dict
          * 'mdp_portable'   : dict (from MDPNetwork.to_portable())
    Rules:
      - If 'networkx_env' is provided, no mdp_* may be provided.
      - If 'networkx_env' is not provided and no mdp_* is provided, returns None.
      - If exactly one mdp_* is provided, build MDPNetwork and wrap as NetworkXMDPEnvironment.
    """
    nx_env = cfg.get("networkx_env", None)

    has_path = "mdp_config_path" in cfg and cfg["mdp_config_path"] is not None
    has_data = "mdp_config_data" in cfg and cfg["mdp_config_data"] is not None
    has_portable = "mdp_portable" in cfg and cfg["mdp_portable"] is not None
    mdp_sources = int(has_path) + int(has_data) + int(has_portable)

    if nx_env is not None:
        if mdp_sources > 0:
            raise ValueError(
                "Provide either 'networkx_env' OR one mdp_* source, not both."
            )
        # trust caller's object
        return nx_env

    if mdp_sources == 0:
        # no networkx_env and no mdp_* -> caller may intend to create a native env
        return None
    if mdp_sources > 1:
        raise ValueError(
            "Provide exactly one of 'mdp_config_path' | 'mdp_config_data' | 'mdp_portable'."
        )

    # Exactly one mdp_* is provided -> build MDPNetwork, then NetworkX env
    if has_path:
        mdp = MDPNetwork(config_path=cfg["mdp_config_path"])
    elif has_data:
        if not isinstance(cfg["mdp_config_data"], dict):
            raise ValueError("'mdp_config_data' must be a dict.")
        mdp = MDPNetwork(config_data=cfg["mdp_config_data"])
    else:  # has_portable
        if not isinstance(cfg["mdp_portable"], dict):
            raise ValueError("'mdp_portable' must be a dict.")
        mdp = MDPNetwork.from_portable(cfg["mdp_portable"])

    return NetworkXMDPEnvironment(mdp_network=mdp, render_mode=None, seed=seed)


# ---- unified factory signatures: (seed: int, cfg: Dict[str, Any]) ----

def make_frozenlake(seed: int, cfg: Dict[str, Any]):
    """
    Optional MDP/NetworkX overrides:
      - networkx_env OR one mdp_* key (see _resolve_networkx_env_or_none).
    If none provided, falls back to native FrozenLake with map params.
    """
    max_steps = int(cfg.get("max_steps", 500))
    nx_env = _resolve_networkx_env_or_none(seed, cfg)

    map_name = cfg.get("map_name", "8x8")
    is_slippery = bool(cfg.get("is_slippery", True))

    env = CustomisedFrozenLakeEnv(
        map_name=map_name,
        is_slippery=is_slippery,
        networkx_env=nx_env,
        render_mode="rgb_array",
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_taxi(seed: int, cfg: Dict[str, Any]):
    """
    Optional MDP/NetworkX overrides:
      - networkx_env OR one mdp_* key (see _resolve_networkx_env_or_none).
    If none provided, falls back to native Taxi params.
    """
    max_steps = int(cfg.get("max_steps", 250))
    nx_env = _resolve_networkx_env_or_none(seed, cfg)

    is_rainy = bool(cfg.get("is_rainy", False))
    fickle_passenger = bool(cfg.get("fickle_passenger", False))

    env = CustomisedTaxiEnv(
        is_rainy=is_rainy,
        fickle_passenger=fickle_passenger,
        networkx_env=nx_env,
        render_mode="rgb_array",
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_minigrid(seed: int, cfg: Dict[str, Any]):
    """
    Optional MDP/NetworkX overrides:
      - networkx_env OR one mdp_* key (see _resolve_networkx_env_or_none).
    If none provided, falls back to native MiniGrid params.
    """
    max_steps = int(cfg.get("max_steps", 1000))
    nx_env = _resolve_networkx_env_or_none(seed, cfg)

    map_name = cfg.get("map_name", "door-key")

    env = CustomMiniGridEnv(
        map_name=map_name,
        random_rotate=False,
        random_flip=False,
        display_mode="middle",
        any_key_opens_the_door=False,
        networkx_env=nx_env,
        render_mode="rgb_array",
    )
    env = IntegerStateObsWrapper(env, keep_raw_obs_in_info=False)
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_nx_env_from_mdp(seed: int, cfg: Dict[str, Any]):
    """Read mdp_config_path from cfg, construct MDP, then wrap."""
    max_steps = int(cfg.get("max_steps", 500))
    nx_env = _resolve_networkx_env_or_none(seed, cfg)

    if nx_env is None:
        # For this factory, we require an explicit NetworkX env or an MDP source.
        raise ValueError(
            "make_nx_env_from_mdp requires either 'networkx_env' or an mdp_* source."
        )

    env = nx_env
    env = TimeLimit(env, max_episode_steps=max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env(env_spec: Dict[str, Any], seed: int):
    """
    env_spec = {
        'factory_path': 'pkg.mod:make_env',   # callable(seed: int, cfg: Dict[str, Any]) -> gym.Env
        'cfg': {...}
    }
    """
    fn = _import(env_spec["factory_path"])
    return fn(seed=seed, cfg=dict(env_spec.get("cfg", {})))
