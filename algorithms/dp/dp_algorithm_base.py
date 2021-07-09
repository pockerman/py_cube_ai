from typing import Any
from algorithms.algorithm_base import AlgorithmBase


class DPAlgoBase(AlgorithmBase):
    """
    Base class for DP-based algorithms
    """

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any, gamma: float) -> None:
        super(DPAlgoBase, self).__init__(n_max_iterations=n_max_iterations,
                                         tolerance=tolerance, env=env)
        self._gamma = gamma

    @property
    def gamma(self) -> float:
        return self._gamma
