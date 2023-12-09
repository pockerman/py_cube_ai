"""Module iterative_policy_evaluation. Implements
a tabular version of the iterative policy evaluation
algorithm as described in the book

http://incompleteideas.net/book/RLbook2020.pdf

"""

import numpy as np
from typing import TypeVar

from pycubeai.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from pycubeai.utils.episode_info import EpisodeInfo
from pycubeai.utils.mixins import WithValueTableMixin
from pycubeai.utils.wrappers import time_func_wrapper

# from pycubeai.algorithms.dp.utils import state_actions_from_v as q_s_a
# from pycubeai.algorithms.dp.utils import q_from_v

Env = TypeVar('Env')


class IterativePolicyEvaluator(DPAlgoBase, WithValueTableMixin):
    """Implements iterative policy evaluation algorithm

    """

    def __init__(self, algo_config: DPAlgoConfig) -> None:
        super(IterativePolicyEvaluator, self).__init__(algo_config)
        self.vtable: np.array = None

    @time_func_wrapper(show_time=False)
    def do_on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to run the training episode
        episode_idx: The episode index
        options: Options that client code may pass

        Returns
        -------

        An instance of EpisodeInfo

        """
        episode_reward = 0.0
        episode_itrs = 0

        delta = 0.0
        # we loop over the states of the environment
        for s in range(env.n_states):

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

        info = EpisodeInfo(episode_reward=episode_reward, episode_iterations=episode_itrs,
                           episode_index=episode_idx)

        if delta <= self.config.tolerance:
            info.info["break_training"] = True

        return info

    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # zero the value function
        self.v = np.zeros(env.n_states)
