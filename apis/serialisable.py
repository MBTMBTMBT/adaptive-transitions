from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="Serialisable")

class Serialisable(ABC):
    """
    ABC for objects that can be serialized to a JSON-friendly dict and
    reconstructed from it. Provides a default clone() implementation.
    """

    @abstractmethod
    def to_portable(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_portable(cls: Type[T], portable: Dict[str, Any]) -> T:
        """Rebuild an instance from the dict produced by to_portable()."""
        raise NotImplementedError

    def clone(self: T) -> T:
        """
        Default clone via round-tripping the portable format.
        Subclasses can override for performance if they want.
        """
        return type(self).from_portable(self.to_portable())
