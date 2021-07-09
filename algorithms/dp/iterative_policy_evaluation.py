from typing import Any
import numpy as np

from algorithms.algorithm_base import AlgorithmBase
from algorithms.dp.dp_algorithm_base import DPAlgoBase
from algorithms.dp.utils import state_actions_from_v as q_s_a
from algorithms.dp.utils import q_from_v
from utils.policies.policy_base import PolicyBase


class IterativePolicyEvaluator(DPAlgoBase):

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any, gamma: float, policy_init: PolicyBase) -> None:
        super(IterativePolicyEvaluator, self).__init__(n_max_iterations=n_max_iterations,
                                                       tolerance=tolerance, env=env, gamma=gamma)

        # 1D numpy array for the value function
        self._v = None
        self._policy = policy_init

    @property
    def v(self) -> np.array:
        return self._v

    @property
    def policy(self) -> PolicyBase:
        return self._policy

    @policy.setter
    def policy(self, value: PolicyBase) -> None:
        self._policy = value

    @property
    def q(self) -> dict:
        """
        Returns the state-action value function for the
        approximated value function
        """
        return q_from_v(env=self.train_env, v=self._v, gamma=self.gamma)

    def state_actions_from_v(self, state: int) -> np.ndarray:
        """
        Given the state index returns the list of actions under the
        established value functions
        """
        return q_s_a(env=self.train_env, v=self._v, gamma=self.gamma, state=state)

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
        #self._policy = self._policy_init()

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