from typing import Any

from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoInput
from src.policies.policy_base import PolicyBase


class Sarsa(TDAlgoBase):
    """
    SARSA algorithm: On-policy TD control.
    Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, algo_in: TDAlgoInput) -> None:

        super().__init__(algo_in=algo_in)
        self._policy = algo_in.policy

    def on_episode(self, **options):
        """
        Perform one step of the algorithm
        """
        score = 0.0
        state = self.train_env.reset()

        # select an action
        action = self._policy(self._q, self.train_env.action_space.n)

        for itr in range(self.n_itrs_per_episode):
            # Take a step
            next_state, reward, done, _ = self.train_env.on_episode(self._action)
            score += reward

            if not done:
                next_action = self._policy(self._q, self.train_env.action_space.n)
                self.update_q_table(next_action=next_action)
                state = next_state
                action = next_action

            if done:
                self.update_q_table(next_action=0)
                self._tmp_scores.append(score)  # append score
                break

            # TD Update
            td_target = reward + self.gamma * self.q_function[next_state][next_action]
            td_delta = td_target - self.q_function[self._state][self._action]
            self.q_function[self._state][self._action] += self.alpha * td_delta

            if done:
                break

            self._action = next_action
            self._state = next_state

    def update_q_table(self, next_action: int) -> None:
        pass






