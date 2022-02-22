from typing import Any, TypeVar

from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from src.utils.mixins import WithQTableMixin
from src.utils import INFO

QTable = TypeVar('QTable')


class Sarsa(TDAlgoBase, WithQTableMixin):
    """
    SARSA algorithm: On-policy TD control.
    Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, algo_in: TDAlgoConfig) -> None:

        super().__init__(algo_in=algo_in)
        self.q_table = {}
        self._policy = algo_in.policy

    @property
    def q_function(self) -> QTable:
        return self.q_table

    def actions_before_training_begins(self, **options) -> None:
        super(Sarsa, self).actions_before_training_begins(**options)

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
        super(Sarsa, self).actions_after_episode_ends()
        self._policy.actions_after_episode(self.current_episode_index)

    def on_episode(self, **options):
        """
        Perform one step of the algorithm
        """
        score = 0.0

        # select an action
        action = self._policy(q_func=self.q_function, state=self.state)

        # dummy counter for how many iterations
        # we actually run. It is used to calculate the
        # average reward per episode
        counter = 0
        for itr in range(self.n_itrs_per_episode):

            # Take a step
            next_state, reward, done, _ = self.train_env.step(action)
            score += reward

            next_action = self._policy(q_func=self.q_function, state=next_state)
            self.update_q_table(reward=reward, current_action=action, next_state=next_state, next_action=next_action)

            action = next_action
            self.state = next_state
            counter += 1

            if done:
                break

        if self.current_episode_index % self.output_msg_frequency == 0:
            print("{0}: On episode {1} training finished with  "
                  "{2} iterations. Total reward={3}".format(INFO, self.current_episode_index, counter, score))

        self.iterations_per_episode.append(counter)
        self.total_rewards[self.current_episode_index] = score

    def update_q_table(self, reward: float, current_action: int, next_state: int, next_action: int) -> None:
        # TD Update
        td_target = reward + self.gamma * self.q_function[next_state, next_action]
        td_delta = td_target - self.q_function[self.state, current_action]
        self.q_function[self.state, current_action] += self.alpha * td_delta






