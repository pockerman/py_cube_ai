from abc import ABC, abstractmethod
from utils.policies.policy_base import PolicyBase

class PolicyAdaptorBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> PolicyBase:
        raise Exception("Must be overridden")