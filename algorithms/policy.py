from abc import abstractmethod
from collections import defaultdict
import numpy as np


class Policy(object):
    """
    Base class for modeling a policy
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, state):
        raise NotImplementedError("The function must be overridden")


class GreedyPolicy(Policy):

    """
    Creates a greedy policy based on Q values.
    """

    def __init__(self, q_table: defaultdict) -> None:
        """
        Constructor.
        q_table: A dictionary that maps from state -> action values
        :param q_table:
        """
        super(GreedyPolicy, self).__init__()
        self._q_table = q_table

    @abstractmethod
    def __call__(self, state) -> np.array:
        """
        Take an observation as input and returns
        a vector of action probabilities.
        """
        A = np.zeros_like(self._q_table[state], dtype=float)
        best_action = np.argmax(self._q_table[state])
        A[best_action] = 1.0
        return A