# simple_pure_tabular_q_agent.py
# English comments only.

from __future__ import annotations
import json, os, tempfile, zipfile
from typing import Optional, Dict, Any, Tuple, Union, Callable, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from simple_agents.apis import Agent, BaseCallback, FunctionCallback, CallbackList
from mdp_network.mdp_tables import QTable


# Assumed available from your codebase:
# - QTable
# - BaseCallback, CallbackList, FunctionCallback
# - Agent (SB3-like)
# - check_for_correct_spaces

class TabularQAgent(Agent):
    """
    Minimal and stable tabular Q-learning agent.
    - Discrete observations (single int) and discrete actions only.
    - Single Q-table backend using your QTable structure.
    - Constant learning rate, no replay buffer, no vectorization tricks.
    - Policy mixture for stochastic prediction: (p_greedy, p_softmax, p_random).
    - Save/Load as a zip containing q_table.json and metadata.json.
    """

    def __init__(
        self,
        env: Union[gym.Env, VecEnv, None],
        *,
        learning_rate: float = 0.5,
        gamma: float = 0.99,
        policy_mix: Tuple[float, float, float] = (0.7, 0.2, 0.1),  # (greedy, softmax, random)
        temperature: float = 1.0,
        seed: Optional[int] = None,
        verbose: int = 1,
    ):
        super().__init__(env)  # wraps to DummyVecEnv if needed
        # Type checks
        if not isinstance(self.observation_space, spaces.Discrete):
            raise TypeError("This agent only supports Discrete observation spaces.")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("This agent only supports Discrete action spaces.")

        self.q = QTable()
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)

        self.policy_mix = self._normalize_mix(policy_mix)
        self.temperature = float(temperature) if temperature > 0 else 1e-6

        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def set_policy_parameters(
        self,
        policy_mix: Tuple[float, float, float],
        temperature: float
    ):
        """
        Update policy mixture and temperature at runtime.

        Args:
            policy_mix: Tuple of 3 floats (p_greedy, p_softmax, p_random)
            temperature: Positive float for softmax exploration
        """
        self.policy_mix = self._normalize_mix(policy_mix)
        self.temperature = float(temperature) if temperature > 0 else 1e-6
        if self.verbose:
            print(f"[TabularQAgent] Updated policy: mix={self.policy_mix}, temperature={self.temperature}")

    # ===== SB3-like core API =====

    def learn(
            self,
            total_timesteps: int,
            callback: Union[None, Callable, List[BaseCallback], BaseCallback] = None,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> "TabularQAgent":
        if reset_num_timesteps:
            self.num_timesteps = 0

        # Prepare callbacks
        callback = self._init_callback(callback)
        locals_ = locals()
        globals_ = globals()
        callback.init_callback(self)
        callback.on_training_start(locals_=locals_, globals_=globals_)

        # VecEnv always
        env: VecEnv = self.env
        n_envs = env.num_envs

        # Reset and cast to int states
        obs = env.reset()
        states = self._obs_to_int_batch(obs)

        # ---- tqdm progress bar ----
        pbar = None
        try:
            if progress_bar:
                try:
                    from tqdm import tqdm
                    # total is the target steps; initial is where we start from
                    pbar = tqdm(
                        total=total_timesteps,
                        initial=self.num_timesteps,
                        desc="Training",
                        unit="step",
                        dynamic_ncols=True,
                        leave=True,
                    )
                except Exception:
                    pbar = None  # silently degrade if tqdm is unavailable

            # Main loop
            while self.num_timesteps < total_timesteps:
                # Select actions per env using stochastic policy mixture
                actions = np.array([self._stochastic_action(s) for s in states], dtype=int)

                next_obs, rewards, dones, infos = env.step(actions)
                next_states = self._obs_to_int_batch(next_obs)

                # Q-learning update per env
                for i in range(n_envs):
                    s = int(states[i])
                    a = int(actions[i])
                    r = float(rewards[i])
                    done = bool(dones[i])
                    s_next = int(next_states[i])

                    old_q = self.q.get_q_value(s, a)
                    max_next = 0.0 if done else self._max_q(s_next)
                    td_target = r + self.gamma * max_next
                    new_q = old_q + self.learning_rate * (td_target - old_q)
                    self.q.set_q_value(s, a, new_q)

                states = next_states
                self.num_timesteps += 1

                # Progress bar update
                if pbar is not None:
                    # Lightweight postfix: show episodes so far
                    pbar.set_postfix_str(f"episodes={callback.n_episodes}")
                    pbar.update(1)

                # Callback on step
                callback.num_timesteps = self.num_timesteps
                if not callback.on_step():
                    if self.verbose:
                        print("[TabularQAgent] Training aborted by callback.")
                    break

                # Episode handling for bookkeeping/callback
                # In VecEnv, 'dones' indicates any env finished; count them
                if np.any(dones):
                    finished = int(np.sum(dones))
                    callback.n_episodes += finished
                    callback.on_rollout_end()

        finally:
            if pbar is not None:
                pbar.close()

        callback.on_training_end()
        return self

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray], int],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ):
        """
        VecEnv-friendly predict:
        - If observation is a batch (np.ndarray of shape [n_envs]), return np.ndarray of actions [n_envs].
        - If observation is a single int, return a single int action.
        - deterministic=True => greedy; else sample by (greedy, softmax, random).
        """
        # batch case: np.ndarray, shape [n_envs]
        if isinstance(observation, np.ndarray):
            obs_arr = observation.reshape(-1).astype(int)
            if deterministic:
                actions = np.array([self._greedy_action(int(s)) for s in obs_arr], dtype=int)
            else:
                actions = np.array([self._stochastic_action(int(s)) for s in obs_arr], dtype=int)
            return actions, state

        # dict obs is not supported for this agent
        if isinstance(observation, dict):
            raise TypeError("Dict observations are not supported by TabularQAgent (expect Discrete int).")

        # single int observation
        s = int(observation)
        if deterministic:
            a = self._greedy_action(s)
        else:
            a = self._stochastic_action(s)
        return a, state

    def save(self, path: str):
        """
        Save as a zip containing:
          - q_table.json: {state: {action: q_value}}
          - metadata.json: agent config (lr, gamma, policy_mix, temperature, spaces)
        """
        # Prepare data
        q_payload = self.q.q_values
        metadata = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "policy_mix": list(self.policy_mix),
            "temperature": self.temperature,
            "observation_space_n": int(self.observation_space.n),
            "action_space_n": int(self.action_space.n),
            "num_timesteps": int(self.num_timesteps),
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            q_path = os.path.join(tmpdir, "q_table.json")
            m_path = os.path.join(tmpdir, "metadata.json")
            with open(q_path, "w") as f:
                json.dump(q_payload, f)
            with open(m_path, "w") as f:
                json.dump(metadata, f, indent=2)

            with zipfile.ZipFile(path, "w") as zf:
                zf.write(q_path, arcname="q_table.json")
                zf.write(m_path, arcname="metadata.json")

        if self.verbose:
            print(f"[TabularQAgent] Saved zip to: {path}")

    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        print_system_info: bool = False,
    ) -> "TabularQAgent":
        """
        Load from a zip containing q_table.json and metadata.json.
        You must provide an env whose spaces match the saved metadata.
        """
        if env is None:
            raise ValueError("env must be provided when loading the agent.")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)

            with open(os.path.join(tmpdir, "metadata.json"), "r") as f:
                meta = json.load(f)
            with open(os.path.join(tmpdir, "q_table.json"), "r") as f:
                qdict = json.load(f)

        # Instantiate with metadata
        agent = cls(
            env=env,
            learning_rate=float(meta["learning_rate"]),
            gamma=float(meta["gamma"]),
            policy_mix=tuple(meta["policy_mix"]),
            temperature=float(meta["temperature"]),
            seed=None,
            verbose=1,
        )
        # Sanity check spaces
        if agent.observation_space.n != int(meta["observation_space_n"]) or \
           agent.action_space.n != int(meta["action_space_n"]):
            raise ValueError("Loaded metadata spaces do not match given env spaces.")

        # Set Q-table
        # ensure int keys
        q_values: Dict[int, Dict[int, float]] = {
            int(s): {int(a): float(v) for a, v in ad.items()} for s, ad in qdict.items()
        }
        agent.q.q_values = q_values
        agent.num_timesteps = int(meta.get("num_timesteps", 0))

        if print_system_info or agent.verbose:
            print(f"[TabularQAgent] Loaded from: {path} | states={len(q_values)}")

        return agent

    # ===== Helpers =====

    def _stochastic_action(self, s: int) -> int:
        # Sample strategy type
        p = self.policy_mix
        u = self.rng.random()
        if u < p[0]:
            return self._greedy_action(s)
        elif u < p[0] + p[1]:
            return self._softmax_action(s, self.temperature)
        else:
            return int(self.rng.integers(self.action_space.n))

    def _greedy_action(self, s: int) -> int:
        nA = self.action_space.n
        qvals = np.array([self.q.get_q_value(s, a) for a in range(nA)], dtype=float)
        max_q = qvals.max()
        best_as = np.flatnonzero(np.isclose(qvals, max_q))
        return int(self.rng.choice(best_as))

    def _softmax_action(self, s: int, temperature: float) -> int:
        nA = self.action_space.n
        qvals = np.array([self.q.get_q_value(s, a) for a in range(nA)], dtype=float)
        qvals -= np.max(qvals)  # numerical stability
        logits = qvals / max(temperature, 1e-8)
        expv = np.exp(logits)
        probs = expv / (np.sum(expv) + 1e-12)
        # if NaN or degenerate, fallback to uniform
        if not np.all(np.isfinite(probs)) or (probs <= 0).all():
            probs = np.ones(nA, dtype=float) / nA
        return int(self.rng.choice(nA, p=probs))

    def _max_q(self, s: int) -> float:
        nA = self.action_space.n
        return max(self.q.get_q_value(s, a) for a in range(nA))

    def _normalize_mix(self, mix: Tuple[float, float, float]) -> Tuple[float, float, float]:
        arr = np.array(mix, dtype=float)
        if np.any(arr < 0):
            raise ValueError("policy_mix must be non-negative.")
        s = arr.sum()
        if s == 0:
            # default to greedy if zero vector
            return (1.0, 0.0, 0.0)
        arr /= s
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    def _obs_to_int_batch(self, obs: Any) -> np.ndarray:
        """
        Convert VecEnv observations to int array of shape [n_envs].
        Supports observations that are already integers or arrays of integers.
        """
        # DummyVecEnv returns np.ndarray of shape [n_envs]
        # If obs is already int array, cast
        obs_arr = np.asarray(obs).reshape(-1)
        return obs_arr.astype(int)

    # Wrap provided callback API styles
    def _init_callback(self, callback):
        if isinstance(callback, list):
            clist = []
            for cb in callback:
                if isinstance(cb, BaseCallback):
                    clist.append(cb)
                elif callable(cb):
                    clist.append(FunctionCallback(cb))
                else:
                    raise ValueError("Unsupported callback type in list!")
            callback = CallbackList(clist)
        elif callable(callback):
            callback = FunctionCallback(callback)
        elif callback is None:
            callback = CallbackList([])
        callback.init_callback(self)
        return callback
