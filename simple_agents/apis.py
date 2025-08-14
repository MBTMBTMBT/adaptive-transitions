from typing import Optional, Union, Callable, Any
from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


class BaseCallback:
    """
    Base class for callbacks (compatible with SB3 style).
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.training_env = None
        self.locals = None
        self.globals = None
        self.num_timesteps = 0
        self.n_calls = 0
        self.n_episodes = 0
        self.model = None

    def init_callback(self, model):
        """
        Initialize callback before training starts.
        """
        self.model = model
        self._init_callback()

    def on_training_start(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def on_step(self):
        """
        Called at each environment step.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_rollout_end(self):
        """
        Called when an episode ends (rollout finished).
        """
        self.n_episodes += 1
        self._on_rollout_end()

    def on_training_end(self):
        """
        Called after the training loop.
        """
        self._on_training_end()

    def _init_callback(
        self,
    ):
        pass

    def _on_training_start(self):
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        pass

    def _on_training_end(self):
        pass


class CallbackList(BaseCallback):
    """
    Combine multiple callbacks into one list callback.
    """

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def _init_callback(self):
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()


class FunctionCallback(BaseCallback):
    """
    Wrap a callable into a BaseCallback.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def on_step(self):
        return self.func(self.model)


class Agent(ABC):
    """
    Abstract base Agent class for tabular methods, similar to Stable-Baselines3 structure.
    Handles both single Gym environments and VecEnvs.
    """

    def __init__(self, env: Union[Env, VecEnv, None]):
        """
        Initialize the Agent.
        :param env: A single environment or a vectorized environment.
        """
        # Automatically wrap a single environment into a DummyVecEnv
        if env and not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        self.env: Optional[VecEnv] = env
        self.observation_space: spaces.Space = self.env.observation_space
        self.action_space: spaces.Space = self.env.action_space
        self.num_timesteps = 0

    def get_env(self) -> Optional[VecEnv]:
        """
        Return the current environment.
        :return: The current VecEnv environment.
        """
        return self.env

    def set_env(self, env: Union[Env, VecEnv]) -> None:
        """
        Set a new environment for the agent. Re-check spaces.
        :param env: A new environment (single or VecEnv).
        """
        # Auto-wrap if it's not a VecEnv
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        # Validate that the new env spaces match the old ones
        check_for_correct_spaces(env, self.observation_space, self.action_space)

        # Set the new env
        self.env = env

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Union[None, Callable, list["BaseCallback"], "BaseCallback"] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "Agent":
        """
        Abstract learn method. Must be implemented by subclasses.
        :param total_timesteps: Number of timesteps for training.
        :param callback: Optional callback function during training.
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Returns self.
        """
        pass

    @abstractmethod
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """
        Predict an action given an observation.
        :param observation: Observation from the environment.
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether to return deterministic actions.
        :return: Action(s) to take.
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        print_system_info: bool = False,
    ):
        """
        Abstract class method to load an agent from a file.

        :param path: Path to the saved agent.
        :param env: Optional environment to load the agent. If None, it will try to load the agent without it.
        :param print_system_info: Whether to print system info when loading.
        :return: An instance of the Agent.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model to the given path.
        Args:
            path: path to save the model to.
        Returns: None
        """
        pass
