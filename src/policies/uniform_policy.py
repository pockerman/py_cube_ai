import numpy as np
from typing import Any

from src.policies.policy_base import PolicyBase


class UniformPolicy(PolicyBase):

    def __init__(self, env: Any, init_val: float = None) -> None:
        super(UniformPolicy, self).__init__(env=env)
        self.init_val = init_val
        self.policy = self.init()

    def init(self) -> np.ndarray:
        if self.init_val is not None:
            return np.ndarray(
                [[self.init_val for _ in range(self.env.action_space.n)] for _ in range(self.env.observation_space.n)])
        return np.ones([self.env.observation_space.n, self.env.action_space.n]) / self.env.action_space.n

    @property
    def values(self):
        return self.policy

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
