"""
Simple implementation of approximate Monte Carlo method.

The initial implementation of the algorithm is from the
Reinforcement Learning In Motion series by Manning Publications
https://livevideo.manning.com/module/56_8_5/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/approximate-monte-carlo-predictions?

"""
from typing import TypeVar
import numpy as np

from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_config import AlgoConfig
from src.utils import INFO

Env = TypeVar("Env")
Policy = TypeVar("Policy")


class ApproxMonteCarloConfig(AlgoConfig):
    def __init__(self):
        super(ApproxMonteCarloConfig, self).__init__()
        self.alpha: float = 0.1
        self.gamma: float = 0.1
        self.policy: Policy = None


class ApproxMonteCarlo(AlgorithmBase):
    """
    Implements approximate Monte Carlo algorithm
    """

    def __init__(self, algo_config: ApproxMonteCarloConfig) -> None:
        super(ApproxMonteCarlo, self).__init__(algo_in=algo_config)

        assert algo_config.train_env.HAS_DISCRETE_STATES, "Environment does not have discrete states"

        self.n_itrs_per_episode: int = algo_config.n_itrs_per_episode
        self.alpha: float = algo_config.alpha
        self.gamma: float = algo_config.gamma
        self.policy: Policy = algo_config.policy
        self.weights = {}

        self.total_rewards: np.array = np.zeros(self.n_episodes)
        self.iterations_per_episode = []
        self.memory = []

    def state_value(self, state: int) -> float:
        """
        Returns the value associated with the given state
        :param state:
        :return:
        """
        return self.weights[state]

    def actions_before_training_begins(self, **options) -> None:
        """
        Any actions to perform before the training begins
        :param options:
        :return:
        """
        super(ApproxMonteCarlo, self).actions_before_training_begins(**options)
        self._init_weights()

    def actions_after_training_ends(self, **options) -> None:
        """
        Any actions after training is finished
        :param options:
        :return:
        """
        pass

    def actions_before_episode_begins(self, **options) -> None:
        """
        Any actions before the episode begins
        :param options:
        :return:
        """
        super(ApproxMonteCarlo, self).actions_before_episode_begins(**options)

        assert self.n_itrs_per_episode != 0, "Episode number of iterations is zero"

        self.memory = []

    def actions_after_episode_ends(self, **options) -> None:
        """
        Any actions after the episode ends
        :param options:
        :return:
        """
        super(ApproxMonteCarlo, self).actions_after_episode_ends(**options)

        dt = options["dt"]
        G = 0
        last = True
        states_returns = []

        for state, action, reward in reversed(self.memory):
            if last:
                last = False
            else:
                states_returns.append((state, G))

            G = self.gamma * G + reward
        states_returns.reverse()
        states_visited = []
        for state, G, in states_returns:
            if state not in states_visited:
                self._update_weights(total_return=G, state=state, t=dt)
                states_visited.append(state)

    def on_episode(self, **options) -> None:
        """
        Algorithm actions on an episode
        :param options:
        :return:
        """

        episode_reward = 0.0
        counter = 0
        reward = 0.0
        action = 0
        for itr in range(self.n_itrs_per_episode):

            action = self.policy(self.state)

            #self.train_env.render()

            next_state, reward, done, info = self.train_env.step(action)
            self.memory.append((self.state, action, reward))
            self.state = next_state
            episode_reward += reward
            counter += 1

            if done:
                print("{0} episode {1} finished at "
                          "iteration {2}. Total reward {3}".format(INFO,
                                                                   self.current_episode_index, itr, episode_reward))
                break

        self.memory.append((self.state, action, reward))
        self.total_rewards[self.current_episode_index] = episode_reward
        self.iterations_per_episode.append(counter)

    def _init_weights(self) -> None:
        for s in self.train_env.discrete_observation_space:
            self.weights[s] = 0

    def _update_weights(self, total_return: float, state: int, t: float) -> None:
        self.weights[state] += self.alpha / t * (total_return - self.state_value(state))

