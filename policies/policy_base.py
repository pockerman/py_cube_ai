"""
Base class for deriving policies
"""

from abc import ABC, abstractmethod
from typing import Any


class PolicyBase(ABC):

    def __init__(self, env: Any) -> None:
        self.env = env

    @property
    @abstractmethod
    def values(self) -> Any:
        raise Exception("Must be overridden")

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        raise Exception("Must be overridden")

    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison. By default returns false
        """
        return False
