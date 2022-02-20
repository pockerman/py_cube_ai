"""
Policy improvement
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

import numpy as np
from typing import TypeVar
from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from src.algorithms.dp.utils import state_actions_from_v as q_s_a

from src.utils.mixins import WithValueTableMixin
from src.utils.episode_info import EpisodeInfo
from src.utils.wrappers import time_fn
from src.worlds.world_helpers import n_states

PolicyAdaptor = TypeVar('PolicyAdaptor')
Env = TypeVar('Env')


class PolicyImprovement(DPAlgoBase, WithValueTableMixin):
    """Implementation of policy improvement
    """

    def __init__(self, algo_in: DPAlgoConfig, v: np.array,  policy_adaptor: PolicyAdaptor) -> None:
        """

        :param algo_in:
        :type algo_in:
        :param v:
        :type v:
        :param policy_adaptor:
        :type policy_adaptor:
        """
        super(PolicyImprovement, self).__init__(algo_config=algo_in)
        self.v = v
        self.policy_adaptor: PolicyAdaptor = policy_adaptor

    @time_fn
    def on_training_episode(self, env: Env, episode_idx: int, **info) -> EpisodeInfo:

        """
         Train the algorithm on the given environment
        Parameters
        ----------
        env The environment to train on
        episode_idx The episode the trainer is currently
        info Any useful info needed for the training

        Returns
        -------

        EpisodeInfo

        """
        episode_reward = 0.0
        episode_itrs = 0

        n_states_ = n_states(env)

        for s in range(n_states_):
            state_actions = q_s_a(env=env, v=self.v, state=s, gamma=self.gamma)
            self.policy = self.policy_adaptor.adapt(self.policy, state_actions=state_actions,  s=s)

        episode_info = EpisodeInfo()
        episode_info.episode_reward = episode_reward
        episode_info.episode_iterations = episode_itrs
        episode_info.info["break_training"] = True
        return episode_info
