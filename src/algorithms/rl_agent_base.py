"""Module rl_agent_base. Specifies the base class for
deriving reinforcement learning agents. Specifically an RL agent
exposes the following functions

- __init__(AgentConfig) -> None
- on_training_episode(env, episode_idx, **info) -> EpisodeInfo
- play(env, Criterion) -> PlayInfo
- on_state(state) -> Action
- actions_before_training_begins(env, episode_idx, **info) -> None
- actions_before_episode_begins(env, episode_idx, **info) -> None
- actions_after_episode_ends(env, episode_idx, **info) -> None
- actions_after_training_ends(env, episode_idx, **info) -> None

"""

import abc
from typing import TypeVar, Any

Env = TypeVar('Env')
EpisodeInfo = TypeVar('EpisodeInfo')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')
State = TypeVar('State')
Config = TypeVar('Config')


class RLAgentBase(metaclass=abc.ABCMeta):
    """
    Base class for deriving RL agent

    """

    def __init__(self) -> None:
        """
        Constructor.

        """

        pass

    @abc.abstractmethod
    def get_configuration(self) -> Config:
        """
        Returns the configuration of the agent

        Returns
        -------

        An instance of the configuration relevant to each agent

        """

    @abc.abstractmethod
    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """
        Train the algorithm on the episode

        Parameters
        ----------
        env The environment to run the training episode
        episode_idx The episode index
        options Options that client code may pass

        Returns
        -------

        An instance of EpisodeInfo

        """

    @abc.abstractmethod
    def play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """
        Play the trained agent on the given environment

        Parameters
        ----------

        env The environment to play on
        criterion Specifies the criteria such that the play stops

        Returns
        -------

        Returns an instance of PlayInfo

        """

    @abc.abstractmethod
    def on_state(self, state) -> Any:
        """
        Retrurns an action on the given state

        Parameters
        ----------
        state The state instance presented to the agent

        Returns
        -------

        An instance of the action type

        """

    @abc.abstractmethod
    def actions_before_training_begins(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs before

        Parameters
        ----------
        env The environment to train on
        episode_idx The episode index
        options Any options passed by the client code

        Returns None
        -------

        None

        """

    @abc.abstractmethod
    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode

        Parameters
        ----------

        env The environment to train on
        episode_idx The episode index
        options Any options passed by the client code

        Returns
        -------

        Returns None

        """

    @abc.abstractmethod
    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode

        Parameters
        ----------
        env The environment to train on
        episode_idx The episode index
        options Any options passed by the client code

        Returns
        -------

        Returns None

        """

    @abc.abstractmethod
    def actions_after_training_ends(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished

        Parameters
        ----------
        env The environment to train on
        episode_idx The episode index
        options Any options passed by the client code

        Returns
        -------

        Returns None

        """


