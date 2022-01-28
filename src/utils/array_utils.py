"""
Various helper functions over arrays
"""

import numpy as np


def build_one_hot_encoding(n: int, pos: int, val: int = 1) -> np.array:
    """
    Create an one-hot encoding vector of size N. The output
    vector will have the value val at the given position
    :param n: The size of the vector
    :param pos: The position index for the non-zero value
    :param val: The value at the non-zero position
    :return:
    """
    one_hot_vec = np.zeros(n)
    one_hot_vec[pos] = val
    return one_hot_vec
