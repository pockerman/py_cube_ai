from abc import ABC
import numpy as np
from typing import Any, TypeVar

from src.utils.exceptions import InvalidParameterValue
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_config import AlgoConfig

QTable = TypeVar('QTable')
Policy = TypeVar("Policy")


class TDAlgoInput(AlgoConfig):
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

        # monitor performance
        self.total_rewards: np.array = np.zeros(self.n_episodes)
        self.iterations_per_episode = []

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

        self.total_rewards = np.zeros(self.n_episodes)
        self.iterations_per_episode = []

    @property
    def avg_rewards(self):
        average_rewards = np.zeros(self.n_episodes)

        for i, item in enumerate(self.iterations_per_episode):
            episode_reward = self.total_rewards[i]
            average_rewards[i] = episode_reward / self.iterations_per_episode[i]

        return average_rewards

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass