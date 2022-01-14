"""
Various mixin classes to use for simplifying  code
"""

import numpy as np
from typing import TypeVar, Any

QTable = TypeVar('QTable')


class WithQTableMixin(object):
    """
    Helper class to associate a q_table with an algorithm
     if this is needed.
    """
    def __init__(self):
        # the table representing the q function
        # client code should choose the type of
        # the table
        self.q_table: QTable = None


class WithMaxActionMixin(object):
    """
    The class WithMaxActionMixin.
    """

    def __init__(self) -> None:
        super(WithMaxActionMixin, self).__init__()

    @staticmethod
    def max_action(q_table: QTable, state: Any, n_actions: int) -> int:
        """
        Return the action index that presents the maximum
        value at the given state
        :param state: state index
        :param n_actions: Total number of actions allowed
        :return: The action that corresponds to the maximum value
        """
        values = np.array(q_table[state, a] for a in range(n_actions))
        action = np.argmax(values)
        return int(action)


class WithDoubleMaxActionMixin(object):
    """
    The class WithDoubleMaxActionMixin
    """

    def __init__(self) -> None:
        super(WithDoubleMaxActionMixin, self).__init__()

    @staticmethod
    def max_action(q1_table: QTable, q2_table: QTable,
                   state: Any, n_actions: int) -> int:
        """
        Returns the max action by averaging the state values from the two tables
        :param q1_table:
        :param q2_table:
        :param state:
        :param n_actions:
        :return:
        """
        values = np.array([q1_table[state, a] + q2_table[state, a]] for a in range(n_actions))
        action = np.argmax(values)
        return int(action)

    @staticmethod
    def one_table_max_action(q_table: QTable, state: Any, n_actions: int) -> int:
        """
        Return the action index that presents the maximum
        value at the given state
        :param q_table:
        :param state:
        :param n_actions:
        :return:
        """
        values = np.array(q_table[state, a] for a in range(n_actions))
        action = np.argmax(values)
        return int(action)


