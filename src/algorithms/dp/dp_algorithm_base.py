"""
Base class for dynamic programming
algorithms
"""
from typing import Any
import numpy as np

from src.algorithms.algorithm_base import AlgorithmBase
from src.policies.policy_base import PolicyBase


class DPAlgoBase(AlgorithmBase):
    """
    Base class for DP-based algorithms
    """

    def __init__(self, n_episodes: int, tolerance: float,
                 env: Any, gamma: float, policy: PolicyBase) -> None:
        super(DPAlgoBase, self).__init__(n_episodes=n_episodes,
                                         tolerance=tolerance, env=env)
        self._gamma = gamma
        self._policy = policy

        # 1D numpy array for the value function
        self._v = None

    @property
    def v(self) -> np.array:
        return self._v

    @v.setter
    def v(self, value: np.array):
        self._v = value

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def policy(self) -> PolicyBase:
        return self._policy

    @policy.setter
    def policy(self, value: PolicyBase) -> None:
        self._policy = value

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(DPAlgoBase, self).actions_before_training_begins(**options)

        # zero the value function
        self._v = np.zeros(self.train_env.observation_space.n)

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass
