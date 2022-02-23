"""the module episode_info. Specifies a helper
class to wrap the result when training
an agent.

"""

from dataclasses import dataclass


@dataclass(init=True, repr=True)
class EpisodeInfo(object):

    episode_reward: float = 0.0
    episode_iterations: int = 0
    info = {}

