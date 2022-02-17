"""
Base class for deriving RL agent trainers
"""
import abc
from typing import TypeVar

Env = TypeVar('Env')
TrainInfo = TypeVar('TrainInfo')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')
State = TypeVar('State')
Action = TypeVar('Action')
Config = TypeVar('Config')


class RLAgentTrainerConfig(object):
    def __init__(self):
        self.n_episodes: int = 0


class RLAgentTrainerBase(metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_configuration(self) -> Config:
        """
        Returns the configuration of the agent
        :return:
        """

    @abc.abstractmethod
    def train(self, env: Env, **options) -> TrainInfo:
        """
        :return:
        :rtype:
        """

    @abc.abstractmethod
    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
        """

    @abc.abstractmethod
    def actions_before_episode_begins(self, env: Env,  **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """

    @abc.abstractmethod
    def actions_after_episode_ends(self, env: Env,  **info) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode
        :param options:
        :return:
        """

    @abc.abstractmethod
    def actions_after_training_ends(self, env: Env,  **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """