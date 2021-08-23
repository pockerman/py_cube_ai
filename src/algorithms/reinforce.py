"""
Reinforce class. Class based implementation of the
REINFORCE algorithm. This implementation is basically a wrapper
of the implementation in: https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient

"""

import collections
import numpy as np
from algorithms.algorithm_base import AlgorithmBase
from policy_estimator import PolicyEstimatorBase
from value_estimator import ValueEstimatorBase


Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class Reinforce(AlgorithmBase):
    """
        REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
        function approximator using policy gradient.

        Args:
            env: OpenAI environment.
            estimator_policy: Policy Function to be optimized
            estimator_value: Value function approximator, used as a baseline
            num_episodes: Number of episodes to run for
            discount_factor: Time-discount factor

        Returns:
            An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """

    def __init__(self, max_num_iterations_per_episode: int,
                 episode_generator, policy_estimator: PolicyEstimatorBase,
                 value_estimator: ValueEstimatorBase,
                 discount_factor: float=1.0) -> None:

        self._discount_factor = discount_factor
        self._max_num_iterations_per_episode = max_num_iterations_per_episode
        self._episode_generator = episode_generator
        self._policy_estimator = policy_estimator
        self._value_estimator = value_estimator
        self._episode = []
        self._state = None

    def actions_before_stepping(self, env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        self._episode = [] #self._episode_generator()
        self._state = env.reset()

    def step(self, env, **options) -> None:

        for itr in range(self._max_num_iterations_per_episode):

            action_probs = self._policy_estimator.get_action_probabilities(state=self._state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            self._episode.append(Transition(state=self._state, action=action,
                                            reward=reward, next_state=next_state,
                                            done=done))

            if done:
                break

            self._state = next_state

        # Go through the episode and make policy updates
        for t, transition in enumerate(self._episode):

                # The return after this timestep
                total_return = sum(self._discount_factor ** i * t.reward for i, t in enumerate(self._episode[t:]))

                # Calculate baseline/advantage
                baseline_value = self._value_estimator.get_baseline(state=transition.state)
                advantage = total_return - baseline_value

                # Update our value estimator
                self._value_estimator.update(transition.state, total_return)

                # Update our policy estimator
                self._policy_estimator.update(transition.state, advantage)
