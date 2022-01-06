"""
TD(0) algorithm
"""

import numpy as np
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoInput


class TDZero(TDAlgoBase):
    """
    Implements TD(0) algorithm
    """
    
    def __init__(self, algo_in: TDAlgoInput) -> None:
        super(TDZero, self).__init__(algo_in=algo_in)

        self.v_function = np.zeros(self.train_env.n_states + 1)
        self.policy = algo_in.policy

    def actions_before_training_begins(self, **options) -> None:
        super(TDZero, self).actions_before_training_begins(**options)
        self.v_function = np.zeros(self.train_env.n_states + 1)

    def on_episode(self, **options) -> None:
        """
        Execute the algorithm on a training episode
        :param options: Any options passed
        :return:
        """

        episode_reward = 0.0
        counter = 0
        for itr in range(self.n_itrs_per_episode):

            # get the action to execute
            action = self.policy(self.state)

            # step in the environment. obs should be
            # digitized
            obs_, reward, done, info = self.train_env.step(action)

            self._update_value_function(state=self.state,
                                        next_state=obs_, reward=reward)
            # update the state
            self.state = obs_
            episode_reward += reward
            counter += 1

            if done:
                break

        # update the average reward the agent received in this
        # episode
        self.avg_rewards[self.current_episode_index] = episode_reward / counter

    def _update_value_function(self, state: int,
                               next_state: int, reward: float) -> None:
        self.v_function[state] = self.v_function[state] \
                                 + self.alpha * (reward + self.gamma * self.v_function[next_state] - self.v_function[state])

