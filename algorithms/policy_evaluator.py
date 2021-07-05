"""
Class PolicyEvaluator. Taken from
"""

import numpy as np


class PolicyEvaluator(object):

    def __init__(self) -> None:
        pass

    def __call__(self, **options) -> np.array:
        """
            Evaluate a policy given an environment and a full description of the environment's dynamics.

            Args:
                policy: [S, A] shaped matrix representing the policy.
                env: OpenAI env. env.P represents the transition probabilities of the environment.
                    env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
                    env.nS is a number of states in the environment.
                    env.nA is a number of actions in the environment.
                theta: We stop evaluation once our value function change is less than theta for all states.
                discount_factor: Gamma discount factor.

            Returns:
                Vector of length env.nS representing the value function.
            """

        env = options["environment"]
        discount_factor = options["discount_factor"]
        policy = options["policy"]
        theta = options["theta"]

        # Start with a random (all 0) value function
        V = np.zeros(env.nS)
        while True:
            delta = 0
            # For each state, perform a "full backup"
            for s in range(env.nS):
                v = 0
                # Look at the possible next actions
                for a, action_prob in enumerate(policy[s]):
                    # For each action, look at the possible next states...
                    for prob, next_state, reward, done in env.P[s][a]:
                        # Calculate the expected value
                        v += action_prob * prob * (reward + discount_factor * V[next_state])
                # How much our value function changed (across any states)
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            # Stop evaluating once our value function change is below a threshold
            if delta < theta:
                break
        return np.array(V)
