import random
import numpy as np
from enum import Enum
from typing import Any

from src.policies.policy_base import PolicyBase


class EpsilonDecreaseOption(Enum):

    NONE = 0
    EXPONENTIAL = 1
    INVERSE_STEP = 2
    CONSTANT_RATE = 3

class EpsilonGreedyPolicy(PolicyBase):

    def __init__(self, env: Any, eps: float,
                 decay_op: EpsilonDecreaseOption,
                 max_eps: float=1.0, min_eps: float = 0.001,
                 epsilon_decay_factor: float=0.01):
        super(EpsilonGreedyPolicy, self).__init__(env=env)
        self._eps = eps
        self._n_actions = env.action_space.n
        self._decay_op = decay_op
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._epsilon_decay_factor = epsilon_decay_factor

    def __call__(self, q_func, state) -> int:
        # select greedy action with probability epsilon
        if random.random() > self._eps:
            return np.argmax(q_func[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self._n_actions))

    @property
    def values(self) ->None:
        raise Exception("Should not call")

    def actions_after_episode(self, episode_idx, **options) -> None:

        if  self._decay_op == EpsilonDecreaseOption.NONE:
            return

        if self._decay_op == EpsilonDecreaseOption.INVERSE_STEP:

            if episode_idx == 0:
                episode_idx = 1

            self._eps = 1.0 / episode_idx

        elif self._decay_op == EpsilonDecreaseOption.EXPONENTIAL:
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * np.exp(-self._epsilon_decay_factor * episode_idx)

        elif self._decay_op == EpsilonDecreaseOption.CONSTANT_RATE:
            self._eps -= self._epsilon_decay_factor

        if self._eps < self._min_eps:
            self._eps = self._min_eps


