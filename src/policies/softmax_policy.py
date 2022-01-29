"""
Implementation of softmax policy
"""
import numpy as np
from typing import TypeVar, Any

QTable = TypeVar("QTable")


class SoftMaxPolicy(object):

    def __init__(self, tau: float, n_actions: int) -> None:
        self.tau = tau
        self.n_actions = n_actions

    def __str__(self) -> str:
        return self.__name__

    def choose_action_index(self, values) -> int:
        """
        Choose an index from the given values
        :param values:
        :return:
        """
        softmax = np.exp(values / self.tau) / np.sum(np.exp(values / self.tau))

        # return the action index by choosing from
        return np.random.choice([a for a in range(self.n_actions)], p=softmax)

    def __call__(self, q_table: QTable, state: Any) -> int:
        action_values = [q_table[state, a] for a in range(self.n_actions)]
        softmax = np.exp(np.array(action_values) / self.tau) / np.sum(np.exp(np.array(action_values) / self.tau))

        # return the action index by choosing from
        return np.random.choice([a for a in range(self.n_actions)], p=softmax)

    def actions_after_episode(self, episode_idx: int, **options) -> None:
        """
        Any actions that we need to perform on the policy once
        the episode has finished
        :param episode_idx:
        :param options:
        :return:
        """
        pass
