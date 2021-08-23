import numpy as np
from collections import defaultdict
from algorithms.policy_sampler import PolicySampler
from algorithms.policy import GreedyPolicy
from algorithms.algorithm_base import AlgorithmBase


class MCControlImportanceSampling(AlgorithmBase):

    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    """

    def __init__(self, n_episodes: int, itrs_per_episode: int,
                 discount_factor: float, behavior_policy: PolicySampler,
                 tolerance: float) -> None:
        super(MCControlImportanceSampling, self).__init__(n_max_iterations=n_episodes, tolerance=tolerance)

        self._itrs_per_episode = itrs_per_episode
        self._discount_factor = discount_factor

        # An episode is an array of (state, action, reward) tuples
        self._episode = []
        self._state = None

        # The final action-value function.
        # A dictionary that maps state -> action values
        self._q = None

        # The cumulative denominator of the weighted importance sampling formula
        # (across all episodes)
        self._c = None
        self._behavior_policy = behavior_policy
        self._target_policy = None

    @property
    def target_policy(self):
        return self._target_policy

    @property
    def q(self):
        return self._q

    def actions_before_stepping(self, env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        self._episode = []
        self._state = env.reset()

    def actions_before_training_iterations(self, env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # The final value function
        self._q = defaultdict(lambda: np.zeros(env.action_space.n))
        self._c = defaultdict(lambda: np.zeros(env.action_space.n))
        self._target_policy = GreedyPolicy(q_table=self._q)

    def step(self, env, **options) -> None:

        for t in range(self._itrs_per_episode):
            probabilities = self._behavior_policy(self._state)
            action = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            next_state, reward, done, _ = env.step(action)
            self._episode.append((self._state, action, reward))
            if done:
                break
            self._state = next_state

        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0

        for t in range(len(self._episode))[::-1]:

            state, action, reward = self._episode[t]

            # Update total reward
            G = self._discount_factor * G + reward

            # Update weighted importance sampling formula denominator
            self._c[state][action] += W

            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            self._q[state][action] += (W / self._c[state][action]) * (G - self._q[state][action])

            # If the action taken by the behavior policy is not the action
            # taken by the target policy the probability will be 0 and we can break
            if action != np.argmax(self._target_policy(state)):
                break
            W = W * 1. / self._behavior_policy(state)[action]

    def create_greedy_policy(self, q: defaultdict):
        """
        Creates a greedy policy based on Q values.

        Args:
            q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            A = np.zeros_like(q[state], dtype=float)
            best_action = np.argmax(q[state])
            A[best_action] = 1.0
            return A

        return policy_fn