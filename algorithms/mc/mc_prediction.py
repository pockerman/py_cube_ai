"""
Monte Carlo prediction algorithm
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""
from collections import defaultdict
from typing import Any
import numpy as np
from algorithms.policy_sampler import PolicySampler
from algorithms.algorithm_base import AlgorithmBase


class MCPrediction(AlgorithmBase):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    """
    def __init__(self, env: Any, n_max_iterations: int, gamma: float,  episode_generator: Any) -> None:

        super(MCPrediction, self).__init__(env=env, n_max_iterations=n_max_iterations, tolerance=1.0e-4)

        self._gamma = gamma
        self._episode_generator = episode_generator
        self._returns_sum = None #defaultdict(lambda: np.zeros(env.action_space.n))
        self._N = None #defaultdict(lambda: np.zeros(env.action_space.n))
        self._q = None #defaultdict(lambda: np.zeros(env.action_space.n))

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def q(self):
        return self._q

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(MCPrediction, self).actions_before_training_iterations(**options)

        self._returns_sum = defaultdict(lambda: np.zeros(self.train_env.action_space.n))
        self._N = defaultdict(lambda: np.zeros(self.train_env.action_space.n))
        self._q = defaultdict(lambda: np.zeros(self.train_env.action_space.n))

    def step(self, **options) -> None:
        """
        Do one step in the iteration space.
        :param options:
        :return:
        """

        # generate an episode
        episode = self._episode_generator()

        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([self.gamma ** i for i in range(len(rewards) + 1)])
        # update the sum of the returns, number of visits, and action-value
        # function estimates for each state-action pair in the episode
        for i, state in enumerate(states):
            self._returns_sum[state][actions[i]] += sum(rewards[i:] * discounts[:-(1 + i)])
            self._N[state][actions[i]] += 1.0
            self._q[state][actions[i]] = self._returns_sum[state][actions[i]] / self._N[state][actions[i]]

    def actions_after_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass






