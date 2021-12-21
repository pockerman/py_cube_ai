from abc import ABC

import numpy as np
from collections import defaultdict, deque
from typing import Any, TypeVar

from src.algorithms.algorithm_base import AlgorithmBase

QTable = TypeVar('QTable')


class TDAlgoBase(AlgorithmBase, ABC):
    """
    Base class for temporal differences algorithms
    """

    def __init__(self, n_episodes: int, tolerance: float,
                 env: Any, gamma: float, alpha: float,
                 n_itrs_per_episode: int, plot_freq=10):
        super(TDAlgoBase, self).__init__(n_episodes=n_episodes,
                                         tolerance=tolerance, env=env)

        self.gamma: float = gamma
        self.alpha: float = alpha
        self.n_itrs_per_episode: int = n_itrs_per_episode
        self._plot_freq = plot_freq

        # A dictionay of 1D arrays. _q[s][a]
        # is the estimated action value that
        # corresponds to state s and action a
        self._q: QTable = None

        # monitor performance
        self._tmp_scores = None
        self._avg_scores = None

        self._current_episode_itr_index: int = 0

    @property
    def q_function(self) -> QTable:
        return self._q

    @property
    def current_episode_itr_index(self):
        return self._current_episode_itr_index

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

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(TDAlgoBase, self).actions_before_training_begins(**options)

        # initialize empty dictionary of arrays
        self._q = defaultdict(lambda: np.zeros(self.train_env.action_space.n))

        # TODO: These should be transferred to the respective example
        self._tmp_scores = deque(maxlen=self._plot_freq)  # deque for keeping track of scores
        self._avg_scores = deque(maxlen=self.n_episodes)  # average scores over every plot_every episodes

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass