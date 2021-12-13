"""
Value iteration algorithm.
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning

"""

import collections
from typing import Any, TypeVar

from src.algorithms.dp.dp_algorithm_base import DPAlgoBase
from src.algorithms.dp.policy_improvement import PolicyImprovement
from src.algorithms.dp.utils import state_actions_from_v as q_s_a
from src.policies.policy_base import PolicyBase
from src.policies.policy_adaptor_base import PolicyAdaptorBase

Env = TypeVar("Env")


class ValueIteration(DPAlgoBase):
    """
    Value iteration algorithm encapsulated into a class
    The algorithm has two similar implementations regulated
    by the train_mode enum. When train_mode = TrainMode.DEFAULT
    the implementation from Sutton & Barto is used. When
    train_mode = TrainMode.STOCHASTIC the algorithm
    will query the environment to sample an action. Establishment
    of the state value table is done after the episodes are finished
    based on the counters accumulated in self._rewards and self._transits.
    Thus, in the later implementation, the environment is not queried
    for the dynamics i.e. self.env.P[state][action] as is done in the
    former implementation.
    """

    def __init__(self, n_episodes: int, tolerance: float,
                 env: Env, gamma: float, policy_init: PolicyBase,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        """
        Constructor
        """
        super(ValueIteration, self).__init__(n_episodes=n_episodes, gamma=gamma,
                                             tolerance=tolerance, env=env, policy=policy_init)

        self._p_imprv = PolicyImprovement(env=env, v=self.v, gamma=gamma,
                                          policy_init=policy_init, policy_adaptor=policy_adaptor)

        self._rewards = collections.defaultdict(float)
        self._transits = collections.defaultdict(collections.Counter)

    def step(self, **options) -> None:
        """
        Do one step .
        """
        delta = 0
        for s in range(self.train_env.observation_space.n):
            v = self.v[s]
            self.v[s] = max(q_s_a(env=self.train_env, v=self.v, state=s, gamma=self.gamma))
            delta = max(delta, abs(self.v[s] - v))

        self.itr_control.residual = delta

        self._p_imprv.v = self.v
        self._p_imprv.step()
        self.policy = self._p_imprv.policy




