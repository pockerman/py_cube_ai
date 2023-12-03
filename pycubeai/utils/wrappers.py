from functools import wraps
from typing import Callable
import time

from pycubeai.utils.iteration_controller import ItrControlResult
from pycubeai.utils import INFO


def time_fn(func: Callable):
    """Simple timing wrapper
    """
    @wraps(func)
    def measure(*args, **kwargs):
        time_start = time.perf_counter()
        result = func(*args, **kwargs)
        time_end = time.perf_counter()

        if isinstance(result, ItrControlResult):
            result.total_time = time_end - time_start

        print("{0} Done. Execution time"
              " {1} secs".format(INFO, time_end - time_start))
        return result
    return measure


def time_func_wrapper(show_time: bool):
    def _time_func(fn: Callable):
        @wraps(fn)
        def _measure(*args, **kwargs):
            time_start = time.perf_counter()
            result = fn(*args, **kwargs)
            time_end = time.perf_counter()
            if show_time:
                print("{0} Done. Execution time {1} secs".format(INFO, time_end - time_start))

            if isinstance(result, ItrControlResult):
                result.total_time = time_end - time_start
                return result

            return result, time_end - time_start
        return _measure
    return _time_func