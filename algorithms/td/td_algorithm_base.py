import numpy as np
from collections import defaultdict, deque
from algorithms.algorithm_base import AlgorithmBase
from typing import Any


class TDAlgoBase(AlgorithmBase):

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any,  gamma: float, alpha: float, plot_freq=10):
        super(TDAlgoBase, self).__init__(n_max_iterations=n_max_iterations,
                                         tolerance=tolerance, env=env)

        self._gamma = gamma
        self._alpha = alpha
        self._plot_freq = plot_freq

        # A dictionay of 1D arrays. _q[s][a]
        # is the estimated action value that
        # corresponds to state s and action a
        self._q = None

        # monitor performance
        self._tmp_scores = None
        self._avg_scores = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def Q(self) -> dict:
        return self._q

    @property
    def tmp_scores(self):
        return self._tmp_scores

    @property
    def avg_scores(self):
        return self._avg_scores

    @property
    def plot_frequency(self) -> int:
        return self._plot_freq

    @plot_frequency.setter
    def plot_frequency(self, value: int) ->None:
        self._plot_freq = value

    def update_tmp_scores(self, value: float)->None:
        self._tmp_scores.append(value)

    def update_avg_scores(self, value: float) -> None:
        self._avg_scores.append(value)

    def __getitem__(self, item: tuple) -> float:
        return self.train_env[item[0]][item[1]]

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(TDAlgoBase, self).actions_before_training_iterations(**options)

        # initialize empty dictionary of arrays
        self._q = defaultdict(lambda: np.zeros(self.train_env.action_space.n))

        self._tmp_scores = deque(maxlen=self._plot_freq)  # deque for keeping track of scores
        self._avg_scores = deque(maxlen=self.n_max_iterations)  # average scores over every plot_every episodes

    def actions_after_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass