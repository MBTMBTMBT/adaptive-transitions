from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

from gymnasium.core import ObsType


class CustomisableStrEnvAbs(ABC):
    @abstractmethod
    def encode_state(self,) -> str:
        pass

    @abstractmethod
    def decode_state(self, state: str) -> Tuple[ObsType, Dict[str, Any]]:
        pass


class CustomisableIntEnvAbs(ABC):
    @abstractmethod
    def encode_state(self,) -> int:
        pass

    @abstractmethod
    def decode_state(self, state: int) -> Tuple[ObsType, Dict[str, Any]]:
        pass
