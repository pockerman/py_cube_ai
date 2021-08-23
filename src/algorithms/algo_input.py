"""
Utility class to establish algorithm input
"""


class AlgoInput(object):

    def __init__(self):
        self._n_max_iterations = 0
        self._tolerance = 0.0
        self._env = None

    @property
    def n_max_itrs(self) -> int:
        return self._n_max_iterations

    @property
    def env(self):
        return self._env

    @property
    def tolerance(self) -> float:
        return self._tolerance