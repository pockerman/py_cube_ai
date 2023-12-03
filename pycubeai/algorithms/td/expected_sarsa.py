from collections import defaultdict
import numpy as np
import random


from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig


class ExpectedSARSA(TDAlgoBase):

    def __init__(self, n_max_iterations: int, tolerance: float,
                 learning_rate: float, epsilon: float=1.0,
                 discount_factor: float=0.6, max_num_iterations_per_episode: int=1000) -> None:
        super(ExpectedSARSA, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance)

        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._max_num_iterations_per_episode = max_num_iterations_per_episode
        self._Q = None

    def on_episode(self, env, **options):
        """
        Perform one step of the algorithm
        """

        for itr in range(self._max_num_iterations_per_episode):
            # Take a step
            next_state, reward, done, _ = env.on_episode(self._action)

            # Pick the next action
            next_action = self._epsilon_greedy(state=next_state, env=env, eps=self._epsilon)

            expected_q = 0
            qmax = np.max(self._Q[next_state, :])
            greedy_actions = 0

            for i in range(env.nA):
                if self._Q[next_state][i] == qmax:
                    greedy_actions += 1

            non_greedy_action_probability = self._epsilon / env.nA
            greedy_action_probability = ((1 - self._epsilon) / greedy_actions) + non_greedy_action_probability

            for i in range(env.nA):
                if self._Q[next_state][i] == qmax:
                    expected_q += self._Q[next_state][i] * greedy_action_probability
                else:
                    expected_q += self._Q[next_state][i] * non_greedy_action_probability

            # Update statistics
            #stats.episode_rewards[i_episode] += reward
            #stats.episode_lengths[i_episode] = t

            # TD Update
            td_target = reward + self._discount_factor * expected_q
            td_delta = td_target - self._Q[self._state][self._action]

            self._Q[self._state][self._action] += self._learning_rate * td_delta

            # If at the end of learning process
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