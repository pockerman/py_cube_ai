"""
A2C algorithm on CartPole-v1 environment.
"""

import gym
import torch
from torch import nn
from torch.nn import functional as F

from pycubeai.algorithms.ac.a2c_base import A2CBase, A2CConfig
from pycubeai.parallel_utils.torch_processes_handler import TorchProcsHandler
from pycubeai.optimization.optimizer_type import OptimizerType
from pycubeai.optimization.pytorch_optimizer_builder import pytorch_optimizer_builder

N_EPISODES = 1000
N_ITRS_PER_EPISODE = 100
N_PROCS = 7
LEARNING_RATE = 1e-4
OPTIMIZER_TYPE = OptimizerType.ADAM


class A2CCartPole(A2CBase):

    def __init__(self, config: A2CConfig):
        super(A2CCartPole, self).__init__(config=config)

        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


def worker(model: A2CCartPole):
    """
    Worker class for each launched process
    :param model:
    :return:
    """

    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    worker_opt = pytorch_optimizer_builder(model.config.opt_type, params=model.parameters(),
                                           **{"learning_rate": LEARNING_RATE})
    worker_opt.zero_grad()
    for i in range(10):
        worker_opt.zero_grad()
        #values, logprobs, rewards = run_episode(worker_env, model)  # B
        #actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)  # C
        #counter.value = counter.value + 1  # D


if __name__ == '__main__':

    # create the master node for the
    # simulation
    config = A2CConfig()
    config.n_episodes = N_EPISODES
    config.n_itrs_per_episode = N_ITRS_PER_EPISODE

    MasterNode = A2CCartPole(config=config)
    #MasterNode.share_memory()

    #procs_handler = TorchProcsHandler(n_procs=N_PROCS)
    #procs_handler.create_and_start(target=worker, args=None)
    #procs_handler.join_and_terminate()

