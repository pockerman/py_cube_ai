

from typing import TypeVar
import numpy as np

from src.algorithms.td. td_algorithm_base import TDAlgoBase


Env = TypeVar('Env')
Action = TypeVar('Action')
Policy = TypeVar('Policy')


class SarsaSemiGrad(TDAlgoBase):

    def __init__(self,  n_episodes: int, tolerance: float,
                 env: Env, gamma: float, alpha: float,
                 n_itrs_per_episode: int, policy: Policy, plot_freq=10) -> None:
        super(SarsaSemiGrad, self).__init__(n_episodes=n_episodes, tolerance=tolerance, env=env,
                                            gamma=gamma, alpha=alpha, n_itrs_per_episode=n_itrs_per_episode,
                                            plot_freq=plot_freq)

        self.weights = np.zeros(self.train_env.n_states, self.train_env.n_actions)
        self.dt = 1.0
        self.policy = policy

    def q_value(self, state_action: Action) -> float:
        return self.weights.dot(state_action)

    def update_weights(self, total_reward: float, state_action: Action, state_action_: Action, t):
        v1 = self.q_value(state_action=state_action)
        v2 = self.q_value(state_action=state_action_)
        self.weights += self.alpha / t * (total_reward + self.gamma*v2 - v1) * state_action

    def actions_before_episode_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return: None
        """

        super(SarsaSemiGrad, self).actions_before_episode_begins(**options)

        if self.current_episode_index % 100 == 0:
            self.dt += 1.0

    def actions_after_episode_ends(self, **options):

        super(SarsaSemiGrad, self).actions_after_episode_ends()
        self.policy.actions_after_episode(self.current_episode_index, **options)

    def step(self, **options) -> None:
        """

        :param options:
        :return:
        """

        for itr in range(self.n_itrs_per_episode):

            self._current_episode_itr_index += 1

            # choose an action at the current state
            action = self.policy(self.q_function, self.state)

            # step in the environment
            obs, reward, done, _ = self.train_env.step(action)

            if done and itr < self.train_env.max_episode_steps:

                val = self.q_value(state_action=self.state)
                self.weights += self.alpha / self.dt * (reward - val) * self.state
                break

            # update current state
            self.state = obs


