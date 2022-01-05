"""
Tabular Q-learning algorithm
"""

import numpy as np
from typing import Any, TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoInput

Env = TypeVar('Env')
Policy = TypeVar('Policy')


class QLearning(TDAlgoBase):
    """
    epsilon-greedy Q-learning algorithm
    """

    def __init__(self, algo_in: TDAlgoInput) -> None:

        super(QLearning, self).__init__(algo_in=algo_in)

        self._policy = algo_in.policy

    def on_episode(self, **options) -> None:
        """
        Perform one step of the algorithm
        """

        # episode score
        score = 0  # initialize score
        state = self.train_env.reset()  # start episode

        for itr in range(self.n_itrs_per_episode):

            if self.render_env:
                self.train_env.render()

            # epsilon-greedy action selection
            action = self._policy(self.q_function, state)
            next_state, reward, done, info = self.train_env.on_episode(action)  # take action A, observe R, S'
            score += reward  # add reward to agent's score
            self._update_Q_table(state, action, reward, next_state)
            state = next_state  # S <- S'
            if done:
                self.update_tmp_scores(score)  # append score
                break

        if self.current_episode_index % self.plot_frequency == 0:
            avg = np.mean(self.tmp_scores)
            print(">>> Train average score {0}".format(avg))
            self.update_avg_scores(avg)

        self._policy.actions_after_episode(self.current_episode_index)

    def _update_Q_table(self, state: int, action: int, reward: float, next_state: int = None) -> None:
        """Update the Q-value for the state"""

        # estimate in Q-table (for current state, action pair)
        current = self.q_function[state][action]
        Qsa_next = np.max(self.q_function[next_state]) if next_state is not None else 0  # value of next state
        target = reward + (self.gamma * Qsa_next)  # construct TD target
        new_value = current + (self.alpha * (target - current))  # get updated value
        self.q_function[state][action] = new_value


