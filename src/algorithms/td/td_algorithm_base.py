from abc import ABC
import numpy as np
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


class WithQTableMixin(object):
    """
    Helper class to associate a q_table with an algorithm
     if this is needed.
    """
    def __init__(self):
        # the table representing the q function
        # client code should choose the type of
        # the table
        self.q_table: QTable = None


class WithMaxActionMixin(WithQTableMixin):

    def __init__(self):
        super(WithMaxActionMixin, self).__init__()

    def max_action(self, state: int, n_actions: int) -> int:
        """
        Return the action index that presents the maximum
        value at the given state
        :param state: state index
        :param n_actions: Total number of actions allowed
        :return: The action that corresponds to the maximum value
        """
        values = np.array(self.q_table[state, a] for a in range(n_actions))
        action = np.argmax(values)
        return int(action)


class TDAlgoBase(AlgorithmBase, ABC):
    """
    Base class for temporal differences algorithms
    """

    def __init__(self, algo_in: TDAlgoInput):
        super(TDAlgoBase, self).__init__(algo_in=algo_in)

        self.gamma: float = algo_in.gamma
        self.alpha: float = algo_in.alpha
        self.n_itrs_per_episode: int = algo_in.n_itrs_per_episode

        # monitor performance
        self.avg_rewards = None

        self.current_episode_itr_index: int = 0

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

        # why a deque and not an array?
        self.avg_rewards = np.zeros(self.n_episodes + 1)

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass