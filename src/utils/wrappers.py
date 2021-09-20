from functools import wraps
import time

from src.utils.iteration_controller import ItrControlResult
from src.utils import INFO


def time_fn(func):
    """
    Simple timing wrapper
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