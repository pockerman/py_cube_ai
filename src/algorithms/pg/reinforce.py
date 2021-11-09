"""
Reinforce class. Class based implementation of the
REINFORCE algorithm. This implementation is basically a wrapper
of the implementation in: https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient

"""

import collections
from collections import deque
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

    def __init__(self, n_max_iterations: int, tolerance: float, env: Any,
                 max_itrs_per_episode: int, policy: PolicyTorchBase) -> None:
        super(Reinforce, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance, env=env)
        self._max_itrs_per_episode = max_itrs_per_episode
        self.scores = deque(maxlen=100)
        self._policy = policy
        self.saved_log_probs = []
        self.rewards = []

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(Reinforce, self).actions_before_training_iterations(**options)
        self.scores = deque(maxlen=100)
        self._reset_internal_structs()

    def _reset_internal_structs(self) -> None:

        self.saved_log_probs = []
        self.rewards = []


    def step(self, **options) -> None:

        # for every episode reset the environment
        self.state = self.train_env.reset()
        self._reset_internal_structs()
        
        for itr in range(self._max_itrs_per_episode):

            action, log_prob = self._policy.act(state=self.state)



