from abc import ABC

import numpy as np
from collections import defaultdict, deque
from typing import Any, TypeVar

from src.utils.exceptions import InvalidParameterValue
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_input import AlgoInput

QTable = TypeVar('QTable')
Policy = TypeVar("Policy")


class TDAlgoInput(AlgoInput):
    def __init__(self) -> None:
        super(TDAlgoInput, self).__init__()
        self.gamma: float = 1.0
        self.alpha = 0.1
        self.n_itrs_per_episode = 100
        self.policy: Policy = None


class TDAlgoBase(AlgorithmBase, ABC):
    """
    Base class for temporal differences algorithms
    """

    def __init__(self, algo_in: TDAlgoInput):
        super(TDAlgoBase, self).__init__(algo_in=algo_in)

        self.gamma: float = algo_in.gamma
        self.alpha: float = algo_in.alpha
        self.n_itrs_per_episode: int = algo_in.n_itrs_per_episode

        # A dictionay of 1D arrays. _q[s][a]
        # is the estimated action value that
        # corresponds to state s and action a
        self._q: QTable = None

        # monitor performance
        #self._tmp_scores = None
        self.avg_rewards = None

        self.current_episode_itr_index: int = 0

    @property
    def q_function(self) -> QTable:
        return self._q

    #@property
    #def tmp_scores(self):
     #   return self._tmp_scores

    #@property
    #def avg_scores(self):
     #   return self.avg_rewards

    #def update_tmp_scores(self, value: float) -> None:
    #    self._tmp_scores.append(value)

    #def update_avg_scores(self, value: float) -> None:
     #   self.avg_rewards.append(value)

    def __getitem__(self, item: tuple) -> float:
        return self.train_env[item[0]][item[1]]

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(TDAlgoBase, self).actions_before_training_begins(**options)

        if self.n_itrs_per_episode == 0:
            raise InvalidParameterValue(param_name="n_itrs_per_episode", param_val=self.n_itrs_per_episode)

        # initialize empty dictionary of arrays
        self._q = defaultdict(lambda: np.zeros(self.train_env.action_space.n))

        # TODO: These should be transferred to the respective example
        #self._tmp_scores = deque(maxlen=self._plot_freq)  # deque for keeping track of scores

        # why a deque and not an array?
        self.avg_rewards = np.zeros(self.n_episodes + 1)

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass