import numpy as np
from typing import Any
from algorithms.algorithm_base import AlgorithmBase
from algorithms.dp.utils import state_actions_from_v as q_s_a
from utils.policies.policy_base import PolicyBase
from utils.policies.policy_adaptor_base import PolicyAdaptorBase


class PolicyImprovement(AlgorithmBase):
    """
    Implementation of policy improvement
    """

    def __init__(self, env: Any, v: np.array, gamma: float, policy_init: PolicyBase,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        super(PolicyImprovement, self).__init__(env=env, tolerance=1.0e-4, n_max_iterations=1)
        self._v = v
        self._gamma = gamma
        self._policy = policy_init
        self._policy_adaptor = policy_adaptor

    @property
    def policy(self) -> Any:
        return self._policy

    @property
    def v(self) -> np.array:
        return self._v

    @v.setter
    def v(self, value: np.array):
        self._v = value

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(PolicyImprovement, self).actions_before_training_iterations(**options)

    def actions_after_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

    def step(self):
        """
        Perform one step of the algorithm
        """

        for s in range(self.train_env.observation_space.n):
            state_actions = q_s_a(env=self.train_env, v=self._v, state=s, gamma=self._gamma)
            self._policy = self._policy_adaptor(s, state_actions, self._policy)

        # set the residual so that we always converge
        self.itr_control.residual = 1.0e-5