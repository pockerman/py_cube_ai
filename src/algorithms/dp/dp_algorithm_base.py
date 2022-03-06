"""The module dp_algorithm_base. Specifies the
base class for dynamic programming algorithms.
"""
import abc
from typing import Any, TypeVar
from dataclasses import dataclass

from src.algorithms.rl_algorithm_base import RLAgentBase
from src.algorithms.algo_config import AlgoConfig
from src.utils.episode_info import EpisodeInfo


Policy = TypeVar("Policy")
Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
PlayInfo = TypeVar('PlayInfo')


@dataclass(init=True, repr=True)
class DPAlgoConfig(AlgoConfig):
    """Data class to wrap configuration parameters for
    Dynamic programming algorithms
    """

    gamma: float = 0.1
    tolerance: float = 1.0e-8
    policy: Policy = None
    

class DPAlgoBase(RLAgentBase):
    """
    Base class for DP-based algorithms
    """

    def __init__(self, algo_config: DPAlgoConfig) -> None:
        """
        Constructor. Initialize the algorithm by passing the configuration
        instance needed.

        Parameters
        ----------
        algo_config Algorithm configuration

        """
        super(DPAlgoBase, self).__init__(algo_config)

    @property
    def gamma(self) -> float:
        """
        Returns the gamma i.e. the discount constant

        Returns
        -------

        Returns the gamma i.e. the discount constant

        """
        return self.config.gamma

    @property
    def policy(self) -> Policy:
        """
        Returns the policy instance

        Returns
        -------

        An instance of the Policy type

        """
        return self.config.policy

    @policy.setter
    def policy(self, value: Policy) -> None:
        self.config.policy = value

    @abc.abstractmethod
    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------
        env: The environment to run the training episode
        episode_idx: The episode index
        options: Any named options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo

        """

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """ Execute any actions the algorithm needs before
        starting the training episodes

        Parameters
        ----------
        env: The environment to train on
        options: Any named options passed by the client code

        Returns
        -------

        None

        """
        pass

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs after
        the iterations are finished

        Parameters
        ----------
        env: The environment to train on
        options: Any named options passed by the client code

        Returns
        -------

        None

        """
        pass

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs before
        starting the episode

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The episode index
        options: Any named options passed by the client code

        Returns
        -------

        None

        """
        super(DPAlgoBase, self).actions_before_episode_begins(env, episode_idx, **options)

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs after
        ending the episode

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The episode index
        options: Any named options passed by the client code

        Returns
        -------

        None

        """
        pass




