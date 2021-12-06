"""
Reinforce class. Class based implementation of the
REINFORCE algorithm. This implementation is basically a wrapper
of the implementation from Udacity Deep RL repository

"""

import collections
from collections import deque
import torch
import numpy as np
from typing import Any
from src.algorithms.algorithm_base import AlgorithmBase
from src.policies.policy_base import PolicyTorchBase


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class Reinforce(AlgorithmBase):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    """

    def __init__(self, n_episodes: int, tolerance: float, env: Any,
                 max_itrs_per_episode: int, gamma: float, print_frequency: int,
                 policy: PolicyTorchBase,
                 optimizer: Any) -> None:
        super(Reinforce, self).__init__(n_episodes=n_episodes, tolerance=tolerance, env=env)
        self._max_itrs_per_episode = max_itrs_per_episode
        self.scores = []
        self.scores_deque = deque(maxlen=100)
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.print_frequency = print_frequency
        self.policy = policy
        self.optimizer = optimizer

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

        for itr in range(self._max_itrs_per_episode):

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
        if current_episode_idx  % self.print_frequency == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(current_episode_idx, np.mean(self.scores_deque)))
        if np.mean(self.scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(current_episode_idx - 100,
                                                                                       np.mean(self.scores_deque)))
            self.itr_control.residual = self.itr_control.residual * 10**-2



