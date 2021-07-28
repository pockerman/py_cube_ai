import numpy as np
import random
from collections import defaultdict
from algorithms.algorithm_base import AlgorithmBase


class Sarsa(AlgorithmBase):
    """
    SARSA algorithm: On-policy TD control.
    Finds the optimal epsilon-greedy policy.
    """

    def __init__(self, episodes: int, itrs_per_episode: int, tol: float,
           discount_factor: float=1.0, alpha: float=0.5, epsilon: float=0.1) -> None:
        super(AlgorithmBase, self).__init__(n_max_iterations=episodes, tolerance=tol)
        self._max_num_iterations_per_episode = itrs_per_episode
        self._discount_factor = discount_factor
        self._alpha = alpha
        self._epsilon = epsilon
        self._state = None
        self._Q = None
        self._policy = None
        self._action = None

    def step(self, env, **options):
        """
        Perform one step of the algorithm
        """

        for itr in range(self._max_num_iterations_per_episode):
            # Take a step
            next_state, reward, done, _ = env.step(self._action)

            # Pick the next action
            next_action = self._epsilon_greedy(state=next_state, env=env, eps=self._epsilon)

            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + self._discount_factor * self._Q[next_state][next_action]
            td_delta = td_target - self._Q[self._state][self._action]
            self._Q[self._state][self._action] += self._alpha * td_delta

            if done:
                break

            self._action = next_action
            self._state = next_state

    def actions_before_stepping(self, env, **options) -> None:
        """
        Any action before the iteration starts
        """
        # Reset the environment and pick the first action
        self._state = env.reset()
        self._action = self._epsilon_greedy(self._state, env=env, eps=self._epsilon)

    def actions_before_training_iterations(self, env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self._Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def _epsilon_greedy(self, state, env, eps):

        """
        Selects epsilon-greedy action for the given state.
        """

        # select greedy action with probability epsilon
        if random.random() > eps:
            return np.argmax(self._Q[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(env.action_space.n))




