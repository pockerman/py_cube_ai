from abc import ABC, abstractmethod
from typing import TypeVar


PolicyBase = TypeVar('PolicyBase')


class PolicyAdaptorBase(ABC):
    """
    Base class for deriving adaptors
    for policies
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, policy: PolicyBase, *args, **kwargs) -> PolicyBase:
        """
        Adapts the given policy
        :param policy: The policy to adapt
        :param args: Any arguments to use in order to adapt the policy
        :param kwargs:
        :return: PolicyBase
        """