"""
Policy improvement
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

import numpy as np

from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoInput
from src.algorithms.dp.utils import state_actions_from_v as q_s_a
from src.policies.policy_adaptor_base import PolicyAdaptorBase


class PolicyImprovement(DPAlgoBase):
    """
    Implementation of policy improvement
    """

    def __init__(self, algo_in: DPAlgoInput, v: np.array,  policy_adaptor: PolicyAdaptorBase) -> None:
        super().__init__(algo_in=algo_in)
        self.v = v
        self._policy_adaptor = policy_adaptor

    def step(self):
        """
        Perform one step of the algorithm
        """

        for s in range(self.train_env.observation_space.n):
            state_actions = q_s_a(env=self.train_env, v=self._v, state=s, gamma=self.gamma)
            self.policy = self._policy_adaptor(s, state_actions, self.policy)

        # set the residual so that we always converge
        self.itr_control.residual = 1.0e-5