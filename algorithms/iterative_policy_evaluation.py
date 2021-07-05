from typing import Any
import numpy as np

from algorithms.algorithm_base import AlgorithmBase


class IterativePolicyEvaluator(AlgorithmBase):

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any, gamma: float, policy_init: Any) -> None:
        super(IterativePolicyEvaluator, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance, env=env)
        self._gamma = gamma
        # 1D numpy array for the value function
        self._v = None
        self._policy = None
        self._policy_init = policy_init

    @property
    def v(self) -> np.array:
        return self._v

    @property
    def gamma(self) -> float:
        return self._gamma

    def actions_after_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # call the base class version
        super(IterativePolicyEvaluator, self).actions_before_training_iterations(**options)

        # reinitialize the policy
        self._policy = self._policy_init()

        # zero the value function
        self._v = np.zeros(self.train_env.observation_space.n)

    def step(self, **options) -> None:
        """
        Do one step of the algorithm
        """

        delta = 0.0
        # we loop over the states of the environment
        for s in range(self.train_env.observation_space.n):

            old_value_s = self.v[s]

            # loop over actions for states. This is
            # the first sum
            value_s = 0.0
            for action, action_prob in enumerate(self._policy[s]):

                # this is the second sum
                for prob, next_state, reward, done in self.train_env.P[s][action]:
                    value_s += action_prob * prob * (reward + self.gamma * self.v[next_state])

            # update the residual
            delta = max(delta, np.abs(old_value_s - value_s))

            # update the value function table
            self.v[s] = value_s
        self.itr_control.residual = delta