"""
a2c_base module. Implements a base class
for A2C like algorithms
"""

import copy
from typing import TypeVar
from torch import nn
from src.optimization.optimizer_type import OptimzerType
from src.algorithms.rl_agent_base import RLAgentBase
from src.parallel_utils.torch_processes_handler import TorchProcsHandler
from src.optimization.pytorch_optimizer_builder import pytorch_optimizer_builder
from src.utils.episode_info import EpisodeInfo
from src.utils.play_info import PlayInfo

MasterNode = TypeVar('MasterNode')
Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
Action = TypeVar('Action')

class A2CConfig(object):
    """
    Configuration parameters for A2C algorithms
    """
    def __init__(self):
        #self.n_episodes: int = 0
        self.n_itrs_per_episode: int = 0
        self.opt_type: OptimzerType = OptimzerType.INVALID
        self.n_procs: int = 1


class A2CNetworkBase(nn.Module):
    """
    Base class for deriving networks for A2C algorithms
    """
    def __init__(self) -> None:
        super(A2CNetworkBase, self).__init__()


def _a2c_worker(env: Env, model, opt_type: OptimzerType, lr: float):

    worker_env = copy.deepcopy(env) #gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = pytorch_optimizer_builder(model.config.opt_type,
                                           params=model.parameters(),
                                           **{"learning_rate": lr})
    worker_opt.zero_grad()
    for i in range(model.config.n_episodes):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, model)  # B
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)  # C
        counter.value = counter.value + 1  # D


class A2CBase(RLAgentBase):

    def __init__(self, model: A2CNetworkBase, config: A2CConfig) -> None:
        super(A2CBase, self).__init__()
        self.model: A2CNetworkBase = model
        self.config: A2CConfig = config

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def get_configuration(self) -> A2CConfig:
        return self.config

    def actions_before_training_begins(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
        """
        # make the PyTorch model to share memory
        self.model.share_memory()

    def on_training_episode(self, env: Env, episode_idx: int, **info) -> EpisodeInfo:
        pass

    def play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """
        Play the trained agent on the given environment
        :param env: The environment to play on
        :param criterion: Specifies the criteria such that the play stops
        :return: PlayInfo
        """
        pass

    def on_state(self, state) -> Action:
        """
        Retrurns an action on the given state
        :param state:
        :return:
        """

    def actions_before_training_begins(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
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

    def actions_after_training_ends(self, env: Env, episode_idx: int, **info) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass


