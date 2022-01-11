"""
Tabular Q-learning algorithm
"""

import numpy as np
from typing import Any, TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoInput
from src.utils.mixins import WithMaxActionMixin
from src.utils import INFO

Env = TypeVar('Env')
Policy = TypeVar('Policy')
QTable = TypeVar('QTable')


class QLearning(TDAlgoBase, WithMaxActionMixin):
    """
    epsilon-greedy Q-learning algorithm
    """

    def __init__(self, algo_in: TDAlgoInput) -> None:

        super(QLearning, self).__init__(algo_in=algo_in)

        self.q_table = {}
        self._policy = algo_in.policy

    @property
    def q_function(self) -> QTable:
        return self.q_table

    def actions_before_training_begins(self, **options) -> None:
        super(QLearning, self).actions_before_training_begins(**options)

        for state in self.train_env.discrete_observation_space:
            for action in range(self.train_env.action_space.n):
                self.q_table[state, action] = 0.0

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        super(QLearning, self).actions_after_episode_ends()
        self._policy.actions_after_episode(self.current_episode_index)

    def on_episode(self, **options) -> None:
        """
        Perform one step of the algorithm
        """

        # episode score
        episode_score = 0  # initialize score
        counter = 0

        for itr in range(self.n_itrs_per_episode):

            if self.render_env:
                self.train_env.render()

            # epsilon-greedy action selection
            action = self._policy(q_func=self.q_function, state=self.state)

            # take action A, observe R, S'
            next_state, reward, done, info = self.train_env.step(action)

            # add reward to agent's score
            episode_score += reward
            self._update_Q_table(self.state, action, reward, next_state)
            self.state = next_state  # S <- S'
            counter += 1

            if done:
                break

        if self.current_episode_index % self.output_msg_frequency == 0:
            print("{0}: On episode {1} training finished with  "
                  "{2} iterations. Total reward={3}".format(INFO, self.current_episode_index, counter, episode_score))

        self.iterations_per_episode.append(counter)
        self.total_rewards[self.current_episode_index] = episode_score

    def _update_Q_table(self, state: int, action: int, reward: float, next_state: int = None) -> None:
        """
        Update the Q-value for the state
        """

        # estimate in Q-table (for current state, action pair)
        q_s = self.q_function[state, action]

        # value of next state
        Qsa_next = \
            self.q_function[next_state, self.max_action(next_state,
                                                        n_actions=self.train_env.action_space.n)] if next_state is not None else 0
        # construct TD target
        target = reward + (self.gamma * Qsa_next)

        # get updated value
        new_value = q_s + (self.alpha * (target - q_s))
        self.q_function[state, action] = new_value


