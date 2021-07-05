from collections import defaultdict
from algorithms.policy_sampler import PolicySampler
from algorithms.algorithm_base import AlgorithmBase


class MCPrediction(AlgorithmBase):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    """
    def __init__(self, episodes: int, itrs_per_episode: int,
                 tol: float, discount_factor: float, policy: PolicySampler) -> None:

        super(MCPrediction, self).__init__(n_max_iterations=episodes, tolerance=tol)
        self._itrs_per_episode = itrs_per_episode
        self._discount_factor = discount_factor
        # An episode is an array of (state, action, reward) tuples
        self._episode = []
        self._state = None
        self._returns_sum = None
        self._returns_count = None
        self._v = None
        self._policy = policy

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

        # Keeps track of sum and count of returns for each state
        # to calculate an average. We could use an array to save all
        # returns (like in the book) but that's memory inefficient.
        self._returns_sum = defaultdict(float)
        self._returns_count = defaultdict(float)

        # The final value function
        self._v = defaultdict(float)

    def step(self, env, **options) -> None:

        for t in range(self._itrs_per_episode):
            action = self._policy(self._state)
            next_state, reward, done, _ = env.step(action)
            self._episode.append((self._state, action, reward))
            if done:
                break
            self._state = next_state

        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in self._episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i, x in enumerate(self._episode) if x[0] == state)

            # Sum up all rewards since the first occurance
            G = sum([x[2] * (self._discount_factor ** i) for i, x in enumerate(self._episode[first_occurence_idx:])])

            # Calculate average return for this state over all sampled episodes
            self._returns_sum[state] += G
            self._returns_count[state] += 1.0
            self._v[state] = self._returns_sum[state] / self._returns_count[state]



