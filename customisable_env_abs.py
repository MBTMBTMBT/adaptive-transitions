from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union, List
from gymnasium.core import ObsType

from mdp_network import MDPNetwork
from networkx_env.networkx_env import NetworkXMDPEnvironment


class CustomisableEnvAbs(ABC):
    def __init__(self, networkx_env: NetworkXMDPEnvironment = None):
        self.networkx_env = networkx_env

    @abstractmethod
    def encode_state(self,) -> int:
        pass

    @abstractmethod
    def decode_state(self, state: int) -> Tuple[ObsType, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_start_states(self,) -> List[int]:
        pass

    @abstractmethod
    def get_mdp_network(self,) -> MDPNetwork:
        pass
