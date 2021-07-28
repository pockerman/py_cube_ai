"""
Implementation of Policy iteration algorithm. In policy
iteration at each step we do one policy evaluation and one policy
improvement.

Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning

"""

import numpy as np
from typing import Any
import copy

from algorithms.algorithm_base import AlgorithmBase
from algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluator
from algorithms.dp.policy_improvement import PolicyImprovement
from utils.policies.policy_base import PolicyBase
from utils.policies.policy_adaptor_base import PolicyAdaptorBase


class PolicyIteration(AlgorithmBase):
    """
    Policy iteration class
    """

    def __init__(self, n_max_iterations: int, tolerance: float, env: Any,
                 gamma: float, n_policy_eval_steps: int, policy_init: PolicyBase,
                 policy_adaptor: PolicyAdaptorBase):
        super(PolicyIteration, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance, env=env)

        self._p_eval = IterativePolicyEvaluator(env=env, n_max_iterations=n_policy_eval_steps,
                                                tolerance=tolerance, gamma=gamma, policy_init=policy_init)

        self._p_imprv = PolicyImprovement(env=env, v=self._p_eval.v, gamma=gamma,
                                          policy_init=policy_init, policy_adaptor=policy_adaptor)

    @property
    def v(self) -> np.array:
        return self._p_imprv.v

    @property
    def policy(self):
        return self._p_eval.policy

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(PolicyIteration, self).actions_before_training_iterations(**options)
        self._p_eval.actions_before_training_iterations(**options)
        self._p_imprv.actions_before_training_iterations(**options)

    def actions_after_training_iterations(self, **options) -> None:
        pass

    def step(self) -> None:

        # make a copy of the policy already obtained
        old_policy = copy.deepcopy(self._p_eval.policy)

        # evaluate the policy
        self._p_eval.train()

        # update the value function to
        # improve for
        self._p_imprv.v = self._p_eval.v

        # improve the policy
        self._p_imprv.train()

        new_policy = self._p_imprv.policy

        # check of the two policies are the same
        if old_policy == new_policy:
            self.itr_control.residual = self.itr_control.tolerance*10**-1

        self._p_eval.policy = new_policy