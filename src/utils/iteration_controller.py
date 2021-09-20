"""
Utility class for controlling iteration
"""

import sys
from src.utils import INFO


class ItrControlResult(object):
    """
    Utility class returned when IterationController
    signals that iterations have finished
    """

    def __init__(self, tol: float, residual: float,
                 n_itrs: int, n_max_itrs: int, n_procs: int = 1) -> None:

        self.tolerance = tol
        self.residual = residual
        self.total_time = -1.0
        self.n_itrs = n_itrs
        self.n_max_itrs = n_max_itrs
        self.n_procs = n_procs

    @property
    def converged(self):
        return self.residual < self.tolerance

    def __str__(self) -> str:
        repr = " Tolernac=  " + str(self.tolerance) + "\n"
        repr += "Residual=  " + str(self.residual) + "\n"
        repr += "Total time=" + str(self.total_time) + "\n"
        repr += "Num itrs=  " + str(self.n_itrs) + "\n"
        repr += "Max itrs=  " + str(self.n_max_itrs) + "\n"
        repr += "Converged= " + str(self.converged) + "\n"
        return repr


class IterationController(object):
    """
    Iteration controller class.
    """

    def __init__(self, tol: float, n_max_itrs: int) -> None:
        self.tolerance = tol
        self._residual = sys.float_info.max
        self.n_max_itrs = n_max_itrs
        self._current_itr = 0

    @property
    def residual(self) -> float:
        """
        Returns the residual
        """
        return self._residual

    @residual.setter
    def residual(self, value: float) -> None:
        """
        Set the residual
        """
        self._residual = value

    @property
    def current_itr_counter(self) -> int:
        """
        Returns the current iteration index
        """
        return self._current_itr

    def reset(self) -> None:
        """
        Reset the the controller to default settings
        """
        self._current_itr = 0
        self._residual = sys.float_info.max

    def continue_itrs(self) -> bool:
        """
        Decide whether the iterations should
        be continued
        """

        if self._residual < self.tolerance:
            print("{0} Converged!! Residual={1}. Tolerance={2}".format(INFO, self.residual, self.tolerance))
            return False

        if self._current_itr >= self.n_max_itrs:
            return False

        self._current_itr += 1
        return True
