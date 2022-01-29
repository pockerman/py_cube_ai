import random
import numpy as np
from enum import Enum
from typing import Any, TypeVar

from src.policies.policy_base import PolicyBase
from src.utils.mixins import WithMaxActionMixin, WithDoubleMaxActionMixin

UserDefinedDecreaseMethod = TypeVar('UserDefinedDecreaseMethod')
QTable = TypeVar("QTable")


class EpsilonDecreaseOption(Enum):
    """
    Enumeration of methods to decrease epsilon in a greedy policy
    """

    NONE = 0
    EXPONENTIAL = 1
    INVERSE_STEP = 2
    CONSTANT_RATE = 3
    USER_DEFINED = 4


class WithEpsilonDecayMixin(object):
    """
    Helper mixin for decaying epsilon
    """
    def __init__(self, decay_op: EpsilonDecreaseOption, eps: float,
                 max_eps: float=1.0, min_eps: float = 0.001, epsilon_decay_factor: float=0.01,
                 user_defined_decrease_method: UserDefinedDecreaseMethod = None):

        self.user_defined_decrease_method: UserDefinedDecreaseMethod = user_defined_decrease_method
        self.decay_op = decay_op
        self.eps = eps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.epsilon_decay_factor = epsilon_decay_factor

    def decay(self, episode_idx: int, **options) -> None:
        """
        Apply actions on the policy after the end of the episode
        :param episode_idx: The episode index
        :param options:
        :return: None
        """

        if self.decay_op == EpsilonDecreaseOption.NONE:
            return

        if self.decay_op == EpsilonDecreaseOption.USER_DEFINED:
            self.eps = self.user_defined_decrease_method(self.eps, episode_idx)

        if self.decay_op == EpsilonDecreaseOption.INVERSE_STEP:

            if episode_idx == 0:
                episode_idx = 1

            self.eps = 1.0 / episode_idx

        elif self.decay_op == EpsilonDecreaseOption.EXPONENTIAL:
            self.eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(-self.epsilon_decay_factor * episode_idx)

        elif self.decay_op == EpsilonDecreaseOption.CONSTANT_RATE:
            self.eps -= self.epsilon_decay_factor

        if self.eps < self.min_eps:
            self.eps = self.min_eps


class EpsilonGreedyPolicy(PolicyBase, WithMaxActionMixin, WithEpsilonDecayMixin):
    """
    The class EpsilonGreedyPolicy. Models epsilon-greedy policy for
    selecting actions using a tabular representation of the state-action
    function.
    """

    def __init__(self,  eps: float,
                 decay_op: EpsilonDecreaseOption,
                 n_actions: int,
                 max_eps: float = 1.0, min_eps: float = 0.001,
                 epsilon_decay_factor: float = 0.01,
                 user_defined_decrease_method: UserDefinedDecreaseMethod = None) -> None:

        super(EpsilonGreedyPolicy, self).__init__()
        self.eps = eps
        self.n_actions = n_actions
        self.decay_op = decay_op
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.epsilon_decay_factor = epsilon_decay_factor
        self.user_defined_decrease_method: UserDefinedDecreaseMethod = user_defined_decrease_method

    def __call__(self, q_func: QTable, state: Any) -> int:
        # select greedy action with probability epsilon
        if random.random() > self.eps:
            return self.max_action(q_func, state=state, n_actions=self.n_actions)
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self.n_actions))

    def choose_action_index(self, values) -> int:
        """
        Choose an index from the given values
        :param values:
        :return:
        """
        # select greedy action with probability epsilon
        if random.random() > self.eps:
            return int(np.argmax(values))
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self.n_actions))

    def actions_after_episode(self, episode_idx: int, **options) -> None:
        """
        Apply actions on the policy after the end of the episode
        :param episode_idx: The episode index
        :param options:
        :return: None
        """
        self.decay(episode_idx=episode_idx)


class EpsilonDoubleGreedyPolicy(PolicyBase, WithDoubleMaxActionMixin, WithEpsilonDecayMixin):
    """
    The class EpsilonDoubleGreedyPolicy. Models epsilon-greedy policy for
    selecting actions using a tabular representation of the state-action
    function. This class is the same as EpsilonGreedyPolicy. The only difference
    is that it uses tow state-action value functions
    """

    def __init__(self, eps: float,
                 decay_op: EpsilonDecreaseOption,
                 n_actions: int,
                 max_eps: float=1.0, min_eps: float = 0.001,
                 epsilon_decay_factor: float=0.01,
                 user_defined_decrease_method: UserDefinedDecreaseMethod = None) -> None:
        super(EpsilonDoubleGreedyPolicy, self).__init__()
        self.eps = eps
        self.n_actions = n_actions
        self.decay_op = decay_op
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.epsilon_decay_factor = epsilon_decay_factor
        self.user_defined_decrease_method: UserDefinedDecreaseMethod = user_defined_decrease_method

    def __call__(self, q1_func: QTable, q2_func: QTable, state: Any) -> int:
        # select greedy action with probability epsilon
        if random.random() > self.eps:
            return self.max_action(q1_func, q2_func, state=state, n_actions=self._n_actions)
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self._n_actions))

    @property
    def values(self) -> None:
        raise Exception("Should not call")

    def actions_after_episode(self, episode_idx: int, **options) -> None:
        """
        Apply actions on the policy after the end of the episode
        :param episode_idx: The episode index
        :param options:
        :return: None
        """
        self.decay(episode_idx=episode_idx)
