from typing import Any
from algorithms.td.td_algorithm_base import TDAlgoBase
from policies.policy_base import PolicyBase


class Sarsa(TDAlgoBase):
    """
    SARSA algorithm: On-policy TD control.
    Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, n_max_iterations: int, tolerance: float,
                 env: Any,  gamma: float, alpha: float,
                 max_num_iterations_per_episode: int, policy: PolicyBase) -> None:

        super().__init__(n_max_iterations=n_max_iterations, tolerance=tolerance,
                         env=env, gamma=gamma, alpha=alpha)
        self._max_num_iterations_per_episode = max_num_iterations_per_episode
        self._policy = policy

    def step(self,  **options):
        """
        Perform one step of the algorithm
        """
        score = 0.0
        state = self.train_env.reset()

        # select an action
        action = self._policy(self._q, self.train_env.action_space.n)

        for itr in range(self._max_num_iterations_per_episode):
            # Take a step
            next_state, reward, done, _ = self.train_env.step(self._action)
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
            td_target = reward + self._discount_factor * self._Q[next_state][next_action]
            td_delta = td_target - self._Q[self._state][self._action]
            self._Q[self._state][self._action] += self._alpha * td_delta

            if done:
                break

            self._action = next_action
            self._state = next_state

    def update_q_table(self, next_action: int) -> None:
        pass






