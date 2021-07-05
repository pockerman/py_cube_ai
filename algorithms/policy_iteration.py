"""
Policy iteration algorithm. Code edited
from
"""

import numpy as np
from algorithms.algorithm_base import AlgorithmBase
from algorithms.policy_evaluator import PolicyEvaluator


class PolicyIteration(AlgorithmBase):

    def __init__(self, environment,  gamma: float, theta: float, policy_evaluator: PolicyEvaluator,
                 n_max_itrs: int =1000, tolerance: float=1.0e-5) -> None:

        super(PolicyIteration, self).__init__(n_max_iterations=n_max_itrs, tolerance=tolerance)
        self._env = environment
        self._gamma = gamma
        self._policy_evaluator = policy_evaluator
        # Start with a random policy
        self._policy = np.ones([self._env.nS, self._env.nA]) / self._env.nA
        self._theta = theta

    def actions_before_training_iterations(self, env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        pass

    def step(self, env, **options) -> None:
        """
        Do one step of the iteration
        :param options:
        :return:
        """

        # Evaluate the current policy
        V = self._policy_evaluator(**{"policy": self._policy,
                                      "environment": env,
                                      "discount_factor": self._gamma,
                                      "theta": self._theta})

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(self._env.nS):
            # The best action we would take under the current policy
            chosen_a = np.argmax(self._policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = self._one_step_lookahead(env=env, state=s, V=V)
            best_a = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            self._policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy.
        # signal convergence
        if policy_stable:
            self.residual = 0.

    def _one_step_lookahead(self, env, state, V) -> np.array:
        """
        Helper function to calculate the value for all action in a given state.

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in self._env.P[state][a]:
                A[a] += prob * (reward + self._gamma * V[next_state])
        return A