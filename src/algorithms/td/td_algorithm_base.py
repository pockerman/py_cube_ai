from abc import ABC
import numpy as np
from typing import Any, TypeVar
from dataclasses import dataclass

from src.algorithms.rl_agent_base import RLAgentBase
from src.utils.exceptions import InvalidParameterValue
from src.algorithms.algo_config import AlgoConfig
from src.utils.play_info import PlayInfo

QTable = TypeVar('QTable')
Policy = TypeVar("Policy")
Env = TypeVar('Env')
Criterion = TypeVar('Criterion')


@dataclass(init=True, repr=True)
class TDAlgoConfig(AlgoConfig):
    """Configuration class for TD like
    algorithms
    """
    gamma: float = 1.0
    alpha: float = 0.1
    policy: Policy = None
    n_itrs_per_episode: int = 100


class TDAlgoBase(RLAgentBase, ABC):
    """Base class for temporal differences algorithms
    """

    def __init__(self, algo_config: TDAlgoConfig):
        super(TDAlgoBase, self).__init__(config=algo_config)

        # monitor performance
        self.total_rewards: np.array = np.zeros(self.config.n_episodes)
        self.iterations_per_episode = []

        self.current_episode_itr_index: int = 0

    @property
    def gamma(self) -> float:
        return self.config.gamma

    @property
    def alpha(self) -> float:
        return self.config.alpha

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs before

        Parameters
        ----------
        env The environment to train on
        options Any options passed by the client code

        Returns
        -------

        None

        """

        # call the base class version
        super(TDAlgoBase, self).actions_before_training_begins(env, **options)

        if self.config.n_episodes == 0:
            raise InvalidParameterValue(param_name="n_episodes", param_val=self.config.n_episodes)

        if self.config.n_itrs_per_episode == 0:
            raise InvalidParameterValue(param_name="n_itrs_per_episode", param_val=self.config.n_itrs_per_episode)

        self.total_rewards = np.zeros(self.config.n_episodes)
        self.iterations_per_episode = []

    @property
    def avg_rewards(self):
        average_rewards = np.zeros(self.total_rewards.shape[0])

        for i, item in enumerate(self.iterations_per_episode):
            episode_reward = self.total_rewards[i]
            average_rewards[i] = episode_reward / self.iterations_per_episode[i]

        return average_rewards

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished

        Parameters
        ----------
        env The environment to train on
        options Any options passed by the client code

        Returns
        -------

        None

        """
        pass

    def play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """
        Play the trained agent on the given environment

        Parameters
        ----------

        env The environment to play on
        criterion Specifies the criteria such that the play stops

        Returns
        -------

        An instance of PlayInfo

        """
        return PlayInfo()

    def on_state(self, state) -> Any:
        """Get an agent specific result e.g. an action or
        the state value

        Parameters
        ----------

        state The state instance presented to the agent

        Returns
        -------

        An agent specific result e.g. an action or
        the state value

        """
        pass