import numpy as np
from typing import TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase

Env = TypeVar("Env")
Policy = TypeVar("Policy")
State = TypeVar("State")


class TDZeroSemiGrad(TDAlgoBase):

    def __init__(self, n_episodes: int, tolerance: float, n_itrs_per_episode: int,
                 env: Env, gamma: float, alpha: float,
                 policy: Policy, plot_freq=10, update_frequency: int=1000) -> None:
        super(TDZeroSemiGrad, self).__init__(n_episodes=n_episodes, tolerance=tolerance,
                                             n_itrs_per_episode=n_itrs_per_episode,
                                             env=env, gamma=gamma, alpha=alpha, plot_freq=plot_freq)

        self.weights = np.zeros(env.n_states())
        self.policy = policy
        self.update_frequency = update_frequency

    def actions_before_training_begins(self, **options) -> None:
        super(TDZeroSemiGrad, self).actions_before_training_begins(**options)
        self._init_weights()

    def step(self,  **options):
        """
        Perform one step of the algorithm
        """

        obs = self.train_env.reset()

        for itr in range(self.n_itrs_per_episode):

            state = self.train_env.get_state(obs)

            # select an action
            action = self.policy(obs[1])

            # Take a step
            next_obs, reward, done, _ = self.train_env.step(action)

            if done:
                break

            next_state  = self.train_env.get_state(next_obs)
            self._update_weights(reward=reward, state=state, next_state=next_state, t=options["t"])

            obs = next_obs

    def get_state_value(self, state: State) -> float:
        return self.weights.dot(state)

    def _update_weights(self, reward, state: State, next_state: State, t: float) -> None:
        state_value = self.get_state_value(state=state)
        next_state_value = self.get_state_value(state=next_state)
        self.weights += self.alpha/t * (reward + self.gamma*next_state_value - state_value) * state

    def _init_weights(self) -> None:
        self.weights = np.zeros(self.train_env.n_states())
