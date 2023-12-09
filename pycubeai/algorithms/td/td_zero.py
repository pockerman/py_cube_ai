"""
TD(0) algorithm
"""

import numpy as np
from typing import Any, TypeVar

from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from pycubeai.utils.wrappers import time_func_wrapper
from pycubeai.utils.time_step import TimeStep
from pycubeai.utils.episode_info import EpisodeInfo


Env = TypeVar('Env')
Policy = TypeVar('Policy')


class TDZero(TDAlgoBase):
    """
    Implements TD(0) algorithm
    """
    
    def __init__(self, algo_config: TDAlgoConfig) -> None:
        super(TDZero, self).__init__(algo_config=algo_config)

        self.v_function: np.array = None
        self.policy = algo_config.policy

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs before

        Parameters
        ----------

        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        super(TDZero, self).actions_before_training_begins(**options)
        self.v_function = np.zeros(env.n_states + 1)

    @time_func_wrapper(show_time=False)
    def do_on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the agent on the environment at the given episode.

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Any options passes by the client code

        Returns
        -------

        An instance of the EpisodeInfo class

        """

        episode_reward = 0.0
        counter = 0
        for itr in range(self.config.n_itrs_per_episode):

            # get the action to execute
            action = self.policy(self.state)

            # step in the environment. obs should be
            # digitized
            time_step: TimeStep = env.step(action)

            self._update_value_function(state=self.state,
                                        next_state=time_step.observation, reward=time_step.reward)
            # update the state
            self.state = time_step.observation
            episode_reward += time_step.reward
            counter += 1

            if time_step.done:
                break

        episode_info = EpisodeInfo(episode_reward=episode_reward, episode_iterations=counter,
                                   episode_index=episode_idx)
        return episode_info

    def _update_value_function(self, state: int,
                               next_state: int, reward: float) -> None:
        self.v_function[state] = self.v_function[state] \
                                 + self.alpha * (reward + self.gamma * self.v_function[next_state] - self.v_function[state])

