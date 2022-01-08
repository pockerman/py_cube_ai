"""
Base class for dynamic programming
algorithms
"""
from typing import Any, TypeVar
import numpy as np

from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_input import AlgoInput


Policy = TypeVar("Policy")


class DPAlgoInput(AlgoInput):
    
    def __init__(self):
        super(DPAlgoInput, self).__init__()
        self.gamma: float = 0.1
        self.policy: Policy = None


class DPAlgoBase(AlgorithmBase):
    """
    Base class for DP-based algorithms
    """

    def __init__(self, algo_in: DPAlgoInput) -> None:
        super(DPAlgoBase, self).__init__(algo_in=algo_in)
        self.gamma: float = algo_in.gamma
        self.policy: Policy = algo_in.policy

        # 1D numpy array for the value function
        self._v = None

    @property
    def v(self) -> np.array:
        return self._v

    @v.setter
    def v(self, value: np.array):
        self._v = value

    @property
    def policy(self) -> Policy:
        return self._policy

    @policy.setter
    def policy(self, value: Policy) -> None:
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
