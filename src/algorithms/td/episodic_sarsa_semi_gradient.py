"""
Semi-gradient SARSA for episodic environments
"""

from typing import TypeVar
import numpy as np

from src.algorithms.td. td_algorithm_base import TDAlgoBase

Env = TypeVar('Env')
Action = TypeVar('Action')
Policy = TypeVar('Policy')


class EpisodicSarsaSemiGrad(TDAlgoBase):

    def __init__(self,  n_episodes: int, tolerance: float,
                 env: Env, gamma: float, alpha: float,
                 n_itrs_per_episode: int, policy: Policy, plot_freq=10) -> None:
        super(EpisodicSarsaSemiGrad, self).__init__(n_episodes=n_episodes, tolerance=tolerance, env=env,
                                                    gamma=gamma, alpha=alpha, n_itrs_per_episode=n_itrs_per_episode,
                                                    plot_freq=plot_freq)

        self.weights = np.zeros((self.train_env.n_states*self.train_env.n_actions))
        self.dt = 1.0
        self.policy = policy
        self.counters = {}

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

        super(EpisodicSarsaSemiGrad, self).actions_before_episode_begins(**options)

        if self.current_episode_index % 100 == 0:
            self.dt += 1.0

        self._current_episode_itr_index = 0

    def actions_after_episode_ends(self, **options):

        super(EpisodicSarsaSemiGrad, self).actions_after_episode_ends()
        self.policy.actions_after_episode(self.current_episode_index, **options)

    def select_action(self, raw_state) -> int:

        # TODO: For epsilon greedy we may not have to calculate constantly
        vals = []
        for a in range(self.train_env.n_actions):
            sa = self.train_env.get_tiled_state(action=a, obs=raw_state)
            vals.append(self.q_value(state_action=sa))

        vals = np.array(vals)

        # choose an action at the current state
        action = self.policy(vals, raw_state)
        return action

    def step(self, **options) -> None:
        """

        :param options:
        :return:
        """

        action = self.select_action(raw_state=self.state)
        for itr in range(self.n_itrs_per_episode):

            self._current_episode_itr_index += 1

            state_action = self.train_env.get_tiled_state(action=action, obs=self.state)

            # step in the environment
            obs, reward, done, _ = self.train_env.step(action)

            if done and itr < self.train_env.get_property(prop_str="_max_episode_steps"): #max_episode_steps:

                val = self.q_value(state_action=state_action)
                self.weights += self.alpha / self.dt * (reward - val) * state_action
                break

            new_action = self.select_action(raw_state=obs) #self.state)
            sa = self.train_env.get_tiled_state(action=new_action, obs=obs)
            self.update_weights(total_reward=reward, state_action=state_action, state_action_=sa, t=self.dt)

            # update current state and action
            self.state = obs
            action = new_action

        self.counters[self.current_episode_index] = self._current_episode_itr_index


