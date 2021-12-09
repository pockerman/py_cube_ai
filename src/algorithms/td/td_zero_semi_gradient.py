import numpy as np
from typing import TypeVar
from src.algorithms.td.td_algorithm_base import  TDAlgoBase

Env = TypeVar("Env")
Policy = TypeVar("Policy")
State = TypeVar("State")


class TDZeroSemiGrad(TDAlgoBase):

    def __init__(self, n_episodes: int, tolerance: float,
                 env: Env, gamma: float, alpha: float,
                 policy: Policy, plot_freq=10) -> None:
        super(TDZeroSemiGrad, self).__init__(n_episodes=n_episodes, tolerance=tolerance,
                                             env=env, gamma=gamma, alpha=alpha, plot_freq=plot_freq)

        self.weights = np.zeros(env.onsrevation_space.n)
        self.policy = policy

    def actions_before_training_begins(self, **options) -> None:
        super(TDZeroSemiGrad, self).actions_before_training_begins(**options)
        self._init_weights()

    def step(self,  **options):
        """
        Perform one step of the algorithm
        """
        score = 0.0
        state = self.train_env.reset()

        # select an action
        action = self.policy(self._q, self.train_env.action_space.n)

        for itr in range(self.n_itrs_per_episode):
            # Take a step
            next_state, reward, done, _ = self.train_env.step(action)
            score += reward

            if not done:
                next_action = self.policy(self._q, self.train_env.action_space.n)
                self.update_q_table(next_action=next_action)
                state = next_state
                action = next_action

            if done:
                self.update_q_table(next_action=0)
                self._tmp_scores.append(score)  # append score
                break

            # TD Update
            td_target = reward + self.gamma * self._Q[next_state][next_action]
            td_delta = td_target - self._Q[self._state][self._action]
            self._Q[self._state][self._action] += self._alpha * td_delta

            if done:
                break

    def get_state_value(self, state: State) -> float:
        return self.weights.dot(state)

    def _update_weights(self, reward, state: State, next_state: State, t: float) -> None:
        state_value = self.get_state_value(state=state)
        next_state_value = self.get_state_value(state=next_state)

        self.weights += self.alpha/t * (reward + self.gamma*next_state_value - state_value) * state

    def _init_weights(self) -> None:

        for state in self.train_env.states:
            self.weights[state] = 0.0