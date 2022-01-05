"""
Iterative policy evaluation
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

import numpy as np

from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoInput
from src.algorithms.dp.utils import state_actions_from_v as q_s_a
from src.algorithms.dp.utils import q_from_v


class IterativePolicyEvaluator(DPAlgoBase):

    def __init__(self, algo_in: DPAlgoInput) -> None:
        super(IterativePolicyEvaluator, self).__init__(algo_in=algo_in)

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

    def on_episode(self, **options) -> None:
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