"""
Reinforce class. Class based implementation of the
REINFORCE algorithm. This implementation is basically a wrapper
of the implementation from Udacity Deep RL repository

"""

import collections
from collections import deque
import torch
import numpy as np
from typing import Any, TypeVar
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_input import AlgoInput
from src.policies.policy_base import PolicyTorchBase


Optimizer = TypeVar("Optimizer")
Policy = TypeVar("Policy")

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReinforceInput(AlgoInput):
    def __init__(self) -> None:
        super(ReinforceInput, self).__init__()
        self.gamma = 1.0
        self.optimizer: Optimizer = None
        self.policy: Policy = None
        self.n_itrs_per_episode = 100
        self.queue_length = 100


# TODO: Remove magic constants from the implementation
class Reinforce(AlgorithmBase):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    """

    def __init__(self, algo_in: ReinforceInput) -> None:
        super(Reinforce, self).__init__(algo_in=algo_in)
        self.n_itrs_per_episode = algo_in.n_itrs_per_episode
        self.scores = []
        self.scores_deque = deque(maxlen=algo_in.queue_length)
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = algo_in.gamma
        #self.print_frequency = print_frequency
        self.policy = algo_in.policy
        self.optimizer = algo_in.optimizer

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(Reinforce, self).actions_before_training_begins(**options)
        self.scores = []
        self._reset_internal_structs()

    def actions_after_training_ends(self, **options) -> None:
        pass

    def _reset_internal_structs(self) -> None:

        self.saved_log_probs = []
        self.rewards = []
        self.scores_deque = deque(maxlen=100)

    def step(self, **options) -> None:

        # for every episode reset the environment
        self.state = self.train_env.reset()
        self._reset_internal_structs()

        for itr in range(self.n_itrs_per_episode):

            action, log_prob = self.policy.act(state=self.state)

            self.saved_log_probs.append(log_prob)
            state, reward, done, _ = self.train_env.step(action)
            self.train_env.render(mode='rgb_array')
            self.rewards.append(reward)
            if done:
                break

        self.scores_deque.append(sum(self.rewards))
        self.scores.append(sum(self.rewards))

        discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, self.rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        current_episode_idx = self.current_episode_index
        if current_episode_idx % self.output_msg_frequency == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(current_episode_idx, np.mean(self.scores_deque)))

        if np.mean(self.scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(current_episode_idx - 100,
                                                                                       np.mean(self.scores_deque)))
            self.itr_control.residual = self.itr_control.residual * 10**-2



