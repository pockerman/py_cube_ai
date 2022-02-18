import numpy as np
from typing import Any

from src.policies.policy_base import PolicyBase


class UniformPolicy(PolicyBase):

    def __init__(self, n_actions: int, n_states: int, init_val: float = None) -> None:
        super(UniformPolicy, self).__init__()
        self.n_actions: int = n_actions
        self.n_states: int = n_states
        self.init_val = init_val
        self.policy = self.init()

    def init(self) -> np.ndarray:
        if self.init_val is not None:
            return np.ndarray(
                [[self.init_val for _ in range(self.n_actions)] for _ in range(self.n_states)])
        return np.ones([self.n_states, self.n_actions]) / self.n_actions

    def __call__(self, *args, **kwargs) -> Any:
        pass

    def __getitem__(self, item: int):
        return self.policy[item]

    def __setitem__(self, item, value) -> None:
        self.policy[item] = value

    def __eq__(self, other: Any) -> bool:
        """
        Equality comparison. Returns true iff
        other is UniformPolicy and all components are equal
        """

        if not isinstance(other, UniformPolicy):
            return False

        if self.policy.shape != other.policy.shape:
            return False

        comparison = self.policy == other.policy
        return comparison.all()
