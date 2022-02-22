"""
Tabular Double Q-learning algorithm. The algorithm can be
found at https://www.researchgate.net/publication/221619239_Double_Q-learning
"""

import numpy as np
from typing import  TypeVar

from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from src.utils.mixins import WithDoubleMaxActionMixin
from src.utils import INFO

Env = TypeVar('Env')
Policy = TypeVar('Policy')


class DoubleQLearning(TDAlgoBase, WithDoubleMaxActionMixin):
    """
    epsilon-greedy Q-learning algorithm
    """

    def __init__(self, algo_in: TDAlgoConfig) -> None:

        super(DoubleQLearning, self).__init__(algo_in=algo_in)

        self._policy = algo_in.policy
        self.q1_table = {}
        self.q2_table = {}

    def actions_before_training_begins(self, **options) -> None:
        super(DoubleQLearning, self).actions_before_training_begins(**options)

        for state in self.train_env.discrete_observation_space:
            for action in range(self.train_env.action_space.n):
                self.q1_table[state, action] = 0.0
                self.q2_table[state, action] = 0.0

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        super(DoubleQLearning, self).actions_after_episode_ends()
        self._policy.actions_after_episode(self.current_episode_index)

    def on_episode(self, **options) -> None:
        """
        Perform one step of the algorithm
        """

        # episode score
        episode_score = 0
        counter = 0

        for itr in range(self.n_itrs_per_episode):

            if self.render_env:
                self.train_env.render()

            # epsilon-greedy action selection
            action = self._policy(self.q1_table, self.q2_table, self.state)

            # take action A, observe R, S'
            next_state, reward, done, info = self.train_env.step(action)
            episode_score += reward  # add reward to agent's score
            self._update_Q_table(self.state, action, reward, next_state)
            self.state = next_state  # S <- S'

            if done:
                break

        if self.current_episode_index % self.output_msg_frequency == 0:
            print("{0}: On episode {1} training finished with  "
                  "{2} iterations. Total reward={3}".format(INFO, self.current_episode_index, counter, episode_score))

        self.iterations_per_episode.append(counter)
        self.total_rewards[self.current_episode_index] = episode_score

    def _update_Q_table(self, state: int, action: int, reward: float, next_state: int = None) -> None:
        """
        Update the Q-value function for the given state when taking the given action.
        The implementation chooses which of the two tables to update using a coin flip
        """

        rand = np.random.random()

        if rand <= 0.5:

            # estimate in Q-table (for current state, action pair)
            q1_s = self.q1_table[state, action]

            max_act = self.one_table_max_action(self.q1_table, next_state,
                                                n_actions=self.train_env.action_space.n)

            # value of next state
            Qsa_next = \
                self.q2_table[next_state, max_act] if next_state is not None else 0

            # construct TD target
            target = reward + (self.gamma * Qsa_next)

            # get updated value
            new_value = q1_s + (self.alpha * (target - q1_s))
            self.q1_table[state, action] = new_value
        else:

            # estimate in Q-table (for current state, action pair)
            q2_s = self.q2_table[state, action]

            max_act = self.one_table_max_action(self.q2_table, next_state,
                                                n_actions=self.train_env.action_space.n)

            # value of next state
            Qsa_next = \
                self.q1_table[next_state, max_act] if next_state is not None else 0

            # construct TD target
            target = reward + (self.gamma * Qsa_next)

            # get updated value
            new_value = q2_s + (self.alpha * (target - q2_s))
            self.q2_table[state, action] = new_value






