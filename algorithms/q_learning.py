"""
Tabular Q-learning algorithm
"""

import numpy as np
import random
import collections
from typing import Tuple
from algorithms.algorithm_base import AlgorithmBase, TrainMode


class QLearning(AlgorithmBase):
    """
    epsilon-greedy Q-learning algorithm
    """

    def __init__(self, env, n_max_iterations: int, tolerance: float,
                 eta: float, epsilon: float = 1.0, max_epsilon: float = 1.0,
                 min_epsilon: float = 0.01, epsilon_decay: float = 0.01,
                 gamma: float = 0.6, max_num_iterations_per_episode: int = 1000,
                 use_decay: bool = True, use_action_with_greedy_method: bool = True,
                 train_mode: TrainMode = TrainMode.DEFAULT) -> None:

        super(QLearning, self).__init__(env=env, n_max_iterations=n_max_iterations,
                                        tolerance=tolerance)
        self._eta = eta
        self._epsilon = epsilon
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._epsilon_decay = epsilon_decay
        self._gamma = gamma
        self._max_num_iterations_per_episode = max_num_iterations_per_episode
        self._use_decay = use_decay
        self._use_action_with_greedy_method = use_action_with_greedy_method
        self._train_mode: TrainMode = train_mode
        self._q_table = collections.defaultdict(float)

    @property
    def values(self):
        return self._q_table

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        self._gamma = value

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value) -> None:
        self._epsilon = value

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, value: float) -> None:
        self._eta = value

    @property
    def t_mode(self) -> TrainMode:
        return self._train_mode

    @property
    def max_n_iters_per_episode(self) -> int:
        return self._max_num_iterations_per_episode

    def step(self, **options) -> None:
        """
        Perform one step of the algorithm
        """

        for itr in range(self.max_n_iters_per_episode):
            is_done = self.one_episode_iteration()

            if is_done:
                break

    def one_episode_iteration(self) -> bool:

        if self.t_mode == TrainMode.DEFAULT:
            # Take a step
            new_state, reward, is_done, _ = self.train_env.step(self.select_action(state=self.state))

            # Pick the next action
            next_action = self.select_action(state=new_state)

            self._value_update(old_state=self.state, action=next_action, reward=reward, new_state=new_state)
            self.state = new_state
            return is_done
        elif self.t_mode == TrainMode.STOCHASTIC:
            action = self.train_env.action_space.sample()
            old_state = self.state
            new_state, reward, is_done, _ = self.train_env.step(action)
            self.state = self.train_env.reset() if is_done else new_state
            self._value_update(old_state=old_state, action=action, reward=reward, new_state=new_state)
            return is_done
        else:
            raise ValueError("Invalid train mode. "
                             "Mode {0} not in [{1}, {2}]".format(self._train_mode,
                                                                 TrainMode.DEFAULT.name, TrainMode.STOCHASTIC.name))

    def best_value_and_action(self, state: int) -> Tuple[float, int]:
        """
        Returns the best action and value corresponding
        to the given state
        """
        best_value, best_action = None, None
        for action in range(self.train_env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    """
    def _step_default(self):

        for itr in range(self.max_n_iters_per_episode):

            # Take a step
            next_state, reward, done, _ = self.env.step(self._action)

            # Pick the next action
            next_action = self.select_action(state=next_state)

            # TD Update
            td_target = reward + self.gamma * self.values[(next_state, next_action)]
            td_delta = td_target - self.values[(self.state, self._action)]
            self.values[(self.state, self._action)] += self.eta * td_delta

            # keep transits and rewards
            self._transits[(self._state, self._action)][next_state] += 1
            self._rewards[(self._state, self._action, next_state)] = reward

            if done:
                break

            self._action = next_action
            self.state = next_state
    """

    def actions_after_stepping(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        starting the iterations
        """
        super(QLearning, self).actions_after_stepping(**options)

        # TODO: Fix this
        episode_counter = self.iteration
        if self._use_decay:
            self._epsilon = self._min_epsilon + (self._max_epsilon - self._min_epsilon) * \
                            np.exp(- self._epsilon_decay * episode_counter)

    def actions_before_stepping(self, **options) -> None:
        """
        Any action before the iteration starts
        """
        pass

    def actions_after_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

    def select_action(self, state):
        """
        Select an action to execute
        """

        if not self._use_action_with_greedy_method:
            return self.best_value_and_action(state=state)[1]
        else:
            return self._epsilon_greedy(state=state)

    def _epsilon_greedy(self, state):
        """
        Selects epsilon-greedy action for the given state.
        """

        # select greedy action with probability epsilon
        if random.random() > self.epsilon:
            return np.argmax(self.values[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self.train_env.action_space.n))

    def _value_update(self, old_state: int, action: int,
                      reward: float, new_state: int):

        best_v, _ = self.best_value_and_action(new_state)
        new_v = reward + self.gamma * best_v
        old_v = self.values[(old_state, action)]
        self.values[(old_state, action)] = old_v * (1 - self.eta) + new_v * self.eta



