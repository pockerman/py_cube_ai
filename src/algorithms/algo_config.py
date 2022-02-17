"""
Utility class to establish algorithm input
"""

from typing import TypeVar

Env = TypeVar('Env')


class AlgoConfig(object):
    """
    The AlgoInput class. Wraps the common input
    that most algorithms use. Concrete algorithms
    can extend this class to accommodate their specific
    input as well
    """

    def __init__(self):
        self.n_itrs_per_episode: int = 0

