"""The module rl_agent_trainer_base. Specifies the
minimum interface that RL trainers should implement
Base class for deriving RL agent trainers. Specifically,
an RL trainer class should expose at least the following API

- __init__(config: Config, agent: Agent) -> None:
- train(self, env: Env, **options) -> TrainInfo:

"""


import abc
from typing import TypeVar
from dataclasses import dataclass

Env = TypeVar('Env')
TrainInfo = TypeVar('TrainInfo')
Config = TypeVar('Config')
Algorithm = TypeVar('Algorithm')


@dataclass
class RLAgentTrainerConfig(object):
    """The RLAgentTrainerConfig class. Basic configuration
    for RL trainers

    """
    n_episodes: int = 0


class RLAgentTrainerBase(metaclass=abc.ABCMeta):
    """The RLAgentTrainerBase class. Specifies the minimum interface
    that RL trainers should implement

    """

    def __init__(self, config: Config, algorithm: Algorithm):
        """Constructor

        Parameters
        ----------
        config The configuration of the trainer
        agent The agent to train

        """
        self.trainer_config: Config = config
        self.algorithm: Algorithm = algorithm

    @abc.abstractmethod
    def train(self, env: Env, **options) -> TrainInfo:
        """Train the agent on the given environment

        Parameters
        ----------
        env The environment to train on
        options Any options that client code should pass

        Returns
        -------

        An instance of the TrainInfo class

        """
