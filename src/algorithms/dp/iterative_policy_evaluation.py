"""
Iterative policy evaluation
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

import numpy as np
from typing import TypeVar

from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from src.utils.episode_info import EpisodeInfo
from src.utils.mixins import WithValueTableMixin
# from src.algorithms.dp.utils import state_actions_from_v as q_s_a
# from src.algorithms.dp.utils import q_from_v

Env = TypeVar('Env')


class IterativePolicyEvaluator(DPAlgoBase, WithValueTableMixin):
    """
    Implements iterative policy evaluation algorithm
    """

    def __init__(self, algo_config: DPAlgoConfig) -> None:
        super(IterativePolicyEvaluator, self).__init__(algo_config)
        self.vtable = None



    '''
    @property
    def q(self) -> dict:
        """
        Returns the state-action value function for the
        approximated value function
        """
        return q_from_v(env=self.train_env, v=self._v, gamma=self.gamma)
    '''

    '''
    def state_actions_from_v(self, state: int) -> np.ndarray:
        """
        Given the state index returns the list of actions under the
        established value functions
        """
        return q_s_a(env=self.train_env, v=self._v, gamma=self.gamma, state=state)
    '''

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """
        Do one step of the algorithm
        """
        episode_reward = 0.0
        episode_itrs = 0

        delta = 0.0
        # we loop over the states of the environment
        for s in range(env.observation_space.n):

            old_value_s = self.v[s]

            # loop over actions for states. This is
            # the first sum
            value_s = 0.0
            for action, action_prob in enumerate(self.config.policy[s]):
                # this is the second sum
                for prob, next_state, reward, done in env.P[s][action]:
                    value_s += action_prob * prob * (reward + self.gamma * self.v[next_state])
                    episode_reward += reward

            episode_itrs += 1
            # update the residual
            delta = max(delta, np.abs(old_value_s - value_s))

            # update the value function table
            self.v[s] = value_s

        info = EpisodeInfo()
        info.episode_reward = episode_reward
        info.episode_iterations = episode_itrs

        if delta <= self.get_configuration().tolerance:
            info.info["break_training"] = True

        return info

    def actions_before_training_begins(self, env: Env, episode_idx: int, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # zero the value function
        self.v = np.zeros(env.observation_space.n)

    def on_state(self, state: int) -> float:
        """
        Retrurns an action on the given state
        :param state:
        :return:
        """
        return self.v[state]
