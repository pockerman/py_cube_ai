"""
Policy improvement
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

import numpy as np
from typing import Any

from algorithms.dp.dp_algorithm_base import DPAlgoBase
from algorithms.dp.utils import state_actions_from_v as q_s_a
from utils.policies.policy_base import PolicyBase
from utils.policies.policy_adaptor_base import PolicyAdaptorBase


class PolicyImprovement(DPAlgoBase):
    """
    Implementation of policy improvement
    """

    def __init__(self, env: Any, v: np.array, gamma: float, policy_init: PolicyBase,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        super(PolicyImprovement, self).__init__(env=env, tolerance=1.0e-4, n_max_iterations=1,
                                                gamma=gamma, policy=policy_init)
        self.v = v
        self._policy_adaptor = policy_adaptor

    def step(self):
        """
        Perform one step of the algorithm
        """

        for s in range(self.train_env.observation_space.n):
            state_actions = q_s_a(env=self.train_env, v=self._v, state=s, gamma=self._gamma)
            self.policy = self._policy_adaptor(s, state_actions, self.policy)

        # set the residual so that we always converge
        self.itr_control.residual = 1.0e-5