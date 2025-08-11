from typing import Tuple, Dict, Any, List
from gymnasium.core import ObsType
import numpy as np

from mdp_network.mdp_network import MDPNetwork
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from customisable_env_abs import CustomisableEnvAbs


ACTION_NAMES = ["Left", "Down", "Right", "Up"]


class CustomisedFrozenLakeEnv(FrozenLakeEnv, CustomisableEnvAbs):
    """
    A customised FrozenLake environment that implements state encoding/decoding
    and can be driven by a NetworkX-backed MDP environment.

    State encoding is the native FrozenLake index:
        s = row * ncol + col
    """

    def __init__(
        self,
        render_mode: str | None = None,
        desc=None,
        map_name: str = "4x4",
        is_slippery: bool = True,
        networkx_env=None,
    ):
        FrozenLakeEnv.__init__(self, render_mode=render_mode, desc=desc, map_name=map_name, is_slippery=is_slippery)
        CustomisableEnvAbs.__init__(self, networkx_env=networkx_env)

    # -------------------------
    # Core overrides
    # -------------------------
    def step(self, action):
        """Optionally delegate transition to an external NetworkX MDP env."""
        if self.networkx_env is not None:
            # Current encoded state
            s = self.encode_state()

            # Sync external env
            self.networkx_env.current_state = s

            # One step in NetworkX env
            sp, r, terminated, truncated, info = self.networkx_env.step(action)

            # Decode back into this env
            obs, decode_info = self.decode_state(int(sp))
            info.update(decode_info)

            # For visual parity with base env
            self.lastaction = int(action)
            if self.render_mode == "human":
                self.render()

            # TimeLimit truncation is handled by wrappers in gym.make(); we keep False here
            return obs, float(r), bool(terminated), bool(truncated), info

        # Fallback to native FrozenLake dynamics
        return super().step(action)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
    ):
        """
        If a NetworkX-backed env is provided, delegate start-state sampling to it,
        then decode that state into this env's internal representation.
        Otherwise, fall back to the native FrozenLake reset.
        """
        if self.networkx_env is not None:
            # Reset external backend RNG and episode
            sp, backend_info = self.networkx_env.reset(seed=seed)
            sp = int(sp)

            # Sync current state for clarity (reset already set it, but keep explicit)
            self.networkx_env.current_state = sp

            # Decode into this env's internal state (and set internal fields)
            obs, decode_info = self.decode_state(sp)

            # Bookkeeping consistent with base env
            self.lastaction = None

            # Optional rendering
            if self.render_mode == "human":
                self.render()

            # Merge info dicts (backend info first, then decode info)
            info = {}
            if isinstance(backend_info, dict):
                info.update(backend_info)
            if isinstance(decode_info, dict):
                info.update(decode_info)
            return obs, info

        # Fallback: native dynamics
        return super().reset(seed=seed, options=options)

    # -------------------------
    # Encode / Decode
    # -------------------------
    def encode_state(self) -> int:
        """Return the current state as integer index (row * ncol + col)."""
        return int(self.s)

    def decode_state(self, state: int) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Force the environment to the given integer state.
        Resets lastaction to None for a clean render.
        """
        if not (0 <= int(state) < self.nrow * self.ncol):
            raise ValueError(f"Invalid state {state}, must be in [0, {self.nrow * self.ncol - 1}]")

        self.s = int(state)
        self.lastaction = None

        # Compose info payload
        r, c = self._rc_from_state(self.s)
        ch = bytes(self.desc[r, c]).decode("utf-8")
        info = {
            "prob": 1.0,
            "row": r,
            "col": c,
            "tile": ch,  # one of 'S','F','H','G'
            "action_names": ACTION_NAMES,
        }

        if self.render_mode == "human":
            self.render()

        return int(self.s), info

    # -------------------------
    # Start states
    # -------------------------
    def get_start_states(self) -> List[int]:
        """
        All indices where tile == 'S'. Matches the environment's initial_state_distrib support.
        """
        starts: List[int] = []
        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.desc[r, c] == b"S":
                    starts.append(self._to_state(r, c))
        # Fallback: if map is malformed (no 'S'), default to 0
        return starts if starts else [0]

    # -------------------------
    # MDP export
    # -------------------------
    def get_mdp_network(self) -> MDPNetwork:
        """
        Build an MDPNetwork from FrozenLakeEnv.P.
        Rewards are taken directly from the underlying env:
          - stepping onto Goal: +1
          - otherwise: 0
        Terminal states are those that transition with done=True (G/H).
        """
        num_states: int = self.observation_space.n
        num_actions: int = self.action_space.n

        # States & starts
        states = list(range(num_states))
        start_states = [s for s, w in enumerate(self.initial_state_distrib) if w > 0.0]

        # Terminal set (any successor marked done)
        terminal_states_set = set()
        for s in range(num_states):
            for a in range(num_actions):
                for p, sp, r, done in self.P[s][a]:
                    if done:
                        terminal_states_set.add(int(sp))
        terminal_states = sorted(terminal_states_set)

        # Transitions: transitions["s"]["a"]["sp"] = {"p": prob, "r": reward}
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        for s in range(num_states):
            s_key = str(s)
            for a in range(num_actions):
                entries = self.P[s][a]
                if not entries:
                    continue
                a_key = str(a)
                # Aggregate (s,a,s') duplicates by summing probs and prob-weighted reward
                accum: Dict[int, Dict[str, float]] = {}
                for p, sp, r, _done in entries:
                    sp = int(sp)
                    acc = accum.setdefault(sp, {"p": 0.0, "r": 0.0})
                    new_p = acc["p"] + float(p)
                    acc["r"] = (acc["r"] * acc["p"] + float(r) * float(p)) / new_p if new_p > 0.0 else float(r)
                    acc["p"] = new_p

                if accum:
                    transitions.setdefault(s_key, {})
                    a_bucket = transitions[s_key].setdefault(a_key, {})
                    for sp, v in accum.items():
                        a_bucket[str(sp)] = {"p": float(v["p"]), "r": float(v["r"])}

        # Tags: label states by agent's tile type
        agent_on_start: List[int] = []
        agent_on_goal: List[int] = []
        agent_on_hole: List[int] = []
        agent_on_frozen: List[int] = []

        for s in range(num_states):
            r, c = self._rc_from_state(s)
            tile = self.desc[r, c]
            if tile == b"S":
                agent_on_start.append(s)
            elif tile == b"G":
                agent_on_goal.append(s)
            elif tile == b"H":
                agent_on_hole.append(s)
            elif tile == b"F":
                agent_on_frozen.append(s)

        tags = {
            "agent_on_start": sorted(agent_on_start),
            "agent_on_goal": sorted(agent_on_goal),
            "agent_on_hole": sorted(agent_on_hole),
            "agent_on_frozen": sorted(agent_on_frozen),
        }

        config = {
            "num_actions": int(num_actions),
            "states": states,
            "start_states": start_states if start_states else [0],
            "terminal_states": terminal_states,
            "default_reward": 0.0,  # non-terminal steps give 0 in FrozenLake
            "transitions": transitions,
            "tags": tags,
        }
        return MDPNetwork(config_data=config)

    # -------------------------
    # Extras (debug helpers)
    # -------------------------
    def get_state_info(self) -> Dict[str, Any]:
        """Return a readable summary of the current state."""
        s = int(self.s)
        r, c = self._rc_from_state(s)
        tile = bytes(self.desc[r, c]).decode("utf-8")
        return {
            "encoded_state": s,
            "row": r,
            "col": c,
            "tile": tile,  # 'S','F','H','G'
            "is_terminal": tile in ("G", "H"),
            "action_names": ACTION_NAMES,
        }

    def is_valid_state(self, state: int) -> bool:
        """Check if state index is in range (grid bounds)."""
        return 0 <= int(state) < self.nrow * self.ncol

    # -------------------------
    # Utilities
    # -------------------------
    def _to_state(self, row: int, col: int) -> int:
        return int(row) * self.ncol + int(col)

    def _rc_from_state(self, s: int) -> Tuple[int, int]:
        return int(s) // self.ncol, int(s) % self.ncol
