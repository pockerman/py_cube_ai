"""
Base class for deriving reinforcement learning agents.
An RL agent exposes the following functions

- __init__(AgentConfig) -> None
- on_training_episode(env, episode_idx, **info) -> EpisodeInfo
- play(env, Criterion) -> ?
- on_state(state) -> Action
- actions_before_training_begins(env, episode_idx, **info) -> None
- actions_before_episode_begins(env, episode_idx, **info) -> None
- actions_after_episode_ends(env, episode_idx, **info) -> None
- actions_after_training_ends(env, episode_idx, **info) -> None

"""

import abc
from typing import TypeVar

Env = TypeVar('Env')
EpisodeInfo = TypeVar('EpisodeInfo')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')
State = TypeVar('State')
Action = TypeVar('Action')
Config = TypeVar('Config')


class RLAgentBase(metaclass=abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_configuration(self) -> Config:
        """
        Returns the configuration of the agent
        :return:
        """

    @abc.abstractmethod
    def on_training_episode(self, env: Env, episode_idx: int, **info) -> EpisodeInfo:
        """
        Train the algorithm on the episode
        :param env: The environment to run the training episode
        :param episode_idx: Episode index
        :param info: info that a  Trainer may pass
        :return: EpisodeInfo
        """

    @abc.abstractmethod
    def  play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """
        Play the trained agent on the given environment
        :param env: The environment to play on
        :param criterion: Specifies the criteria such that the play stops
        :return: PlayInfo
        """

    @abc.abstractmethod
    def on_state(self, state) -> Action:
        """
        Retrurns an action on the given state
        :param state:
        :return:
        """
    @abc.abstractmethod
    def actions_before_training_begins(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
        """

    @abc.abstractmethod
    def actions_before_episode_begins(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """

    @abc.abstractmethod
    def actions_after_episode_ends(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode
        :param options:
        :return:
        """

    @abc.abstractmethod
    def actions_after_training_ends(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """

