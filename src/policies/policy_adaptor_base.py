from abc import ABC, abstractmethod

from src.policies.policy_base import PolicyBase


class PolicyAdaptorBase(ABC):
    """
    Base class for deriving adaptors
    for policies
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> PolicyBase:
        raise Exception("Must be overridden")