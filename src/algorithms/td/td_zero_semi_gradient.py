import numpy as np
from typing import TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig

Env = TypeVar("Env")
Policy = TypeVar("Policy")
State = TypeVar("State")


class TDZeroSemiGrad(TDAlgoBase):

    def __init__(self, algo_in: TDAlgoConfig) -> None:
        super(TDZeroSemiGrad, self).__init__(algo_in=algo_in)

        # make sure we are working on a tiled environment
        assert self.train_env.EPISODIC_CONSTRAINT, "Environment is not episodic"

        self.weights = np.zeros(self.train_env.n_states)
        self.policy = algo_in.policy

    def actions_before_training_begins(self, **options) -> None:
        super(TDZeroSemiGrad, self).actions_before_training_begins(**options)
        self._init_weights()

    def on_episode(self, **options):
        """
        Perform one step of the algorithm
        """

        obs = self.train_env.reset()

        for itr in range(self.n_itrs_per_episode):

            state = self.train_env.get_state(obs)

            # select an action.
            action = self.policy(obs)

            # Take a step
            next_obs, reward, done, _ = self.train_env.on_episode(action)

            if done:
                break

            next_state = self.train_env.get_state(next_obs)
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
