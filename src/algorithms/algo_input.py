"""
Utility class to establish algorithm input
"""

from typing import TypeVar

Env = TypeVar('Env')


class AlgoInput(object):
    """
    The AlgoInput class. Wraps the common input
    that most algorithms use. Concrete algorithms
    can extend this class to accommodate their specific
    input as well
    """

    def __init__(self):
        self.n_episodes: int  = 0
        self.n_itrs_per_episode: int = 0
        self.tolerance: float = 0.0
        self.train_env: Env = None
        self.output_freq: int = None
        self.render_env: bool = False
