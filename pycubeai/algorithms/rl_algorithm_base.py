"""Module rl_agent_base. Specifies the base class for
deriving reinforcement learning agents. Specifically an RL agent
exposes the following functions

- __init__(AgentConfig) -> None
- on_training_episode(env, episode_idx, **info) -> EpisodeInfo

- actions_before_training_begins(env, **info) -> None
- actions_before_episode_begins(env, episode_idx, **info) -> None
- actions_after_episode_ends(env, episode_idx, **info) -> None
- actions_after_training_ends(env, **info) -> None

"""

import abc
from typing import TypeVar, Any
from pycubeai.utils.wrappers import time_func_wrapper

Env = TypeVar('Env')
EpisodeInfo = TypeVar('EpisodeInfo')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')
State = TypeVar('State')
Config = TypeVar('Config')
State = TypeVar('State')


class RLAgentBase(metaclass=abc.ABCMeta):
    """Base class for deriving RL agent
    """

    def __init__(self, config: Config) -> None:
        """Constructor. Initialize the agent with the configuration
        instance needed

        Parameters
        ----------

        config: The configuration of the agent

        """

        self.config = config
        self.state: State = None

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------
        env: The environment to run the training episode
        episode_idx: The episode index
        options: Options that client code may pass

        Returns
        -------

        An instance of EpisodeInfo

        """
        episode_info, total_time = self.do_on_training_episode(env, episode_idx, **options)
        episode_info.episode_index = episode_idx
        episode_info.total_execution_time = total_time
        return episode_info

    @abc.abstractmethod
    @time_func_wrapper(show_time=True)
    def do_on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to run the training episode
        episode_idx: The episode index
        options: Options that client code may pass

        Returns
        -------

        An instance of EpisodeInfo

        """

    @abc.abstractmethod
    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """Execute any actions the algorithm needs before

        Parameters
        ----------
        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs before
        starting the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        self.state = env.reset()

    @abc.abstractmethod
    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs after
        ending the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------

        None

        """

    @abc.abstractmethod
    def actions_after_training_ends(self, env: Env,  **options) -> None:
        """Execute any actions the algorithm needs after
        the iterations are finished

        Parameters
        ----------
        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """


