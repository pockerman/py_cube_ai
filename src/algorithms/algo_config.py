"""The algo_config module. Specifies a configuration
class for that is used as an input to RL algorithms

"""

from typing import TypeVar
from dataclasses import dataclass

Env = TypeVar('Env')


@dataclass(init=True, repr=True)
class AlgoConfig(object):
    """
    The AlgoConfig class. Wraps the common input
    that most algorithms use. Concrete algorithms
    can extend this class to accommodate their specific
    input as well
    """

    n_episodes: int = 0
    tolerance: float = 1.0e-8
    render_env: bool = False
    render_env_freq: int = -1

