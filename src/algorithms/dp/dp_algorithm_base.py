"""
Base class for dynamic programming
algorithms
"""
import abc
from typing import Any, TypeVar

from src.algorithms.rl_agent_base import RLAgentBase
from src.algorithms.algo_config import AlgoConfig
from src.utils.episode_info import EpisodeInfo


Policy = TypeVar("Policy")
Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')


class DPAlgoConfig(AlgoConfig):
    
    def __init__(self):
        super(DPAlgoConfig, self).__init__()
        self.gamma: float = 0.1
        self.tolerance: float = 1.0e-8
        self.policy: Policy = None


class DPAlgoBase(RLAgentBase):
    """
    Base class for DP-based algorithms
    """

    def __init__(self, algo_config: DPAlgoConfig) -> None:
        super(DPAlgoBase, self).__init__()
        self.config = algo_config

    @property
    def gamma(self) -> float:
        return self.config.gamma

    @property
    def policy(self) -> Policy:
        return self.config.policy

    @policy.setter
    def policy(self, value: Policy) -> None:
        self.config.policy = value

    def get_configuration(self) -> DPAlgoConfig:
        """
        Returns the configuration of the agent
        :return:
        """
        return self.config

    @abc.abstractmethod
    def on_training_episode(self, env: Env, episode_idx: int, **info) -> EpisodeInfo:
        """
        Train the algorithm on the episode
        :param env: The environment to run the training episode
        :param episode_idx: Episode index
        :param info: info that a  Trainer may pass
        :return: EpisodeInfo
        """
        pass

    def actions_before_training_begins(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        pass

    def actions_after_training_ends(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        pass

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode
        :param options:
        :return:
        """
        pass

    def play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """
        Play the trained agent on the given environment
        :param env: The environment to play on
        :param criterion: Specifies the criteria such that the play stops
        :return: PlayInfo
        """
        pass

    def on_state(self, state: int) -> Any:
        """
        Retrurns an action on the given state
        :param state:
        :return:
        """
        pass

