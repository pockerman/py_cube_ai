"""the module episode_info. Specifies a helper
class to wrap the result returned by on_training_episode
of a reinforcement learning algorithm


"""

from dataclasses import dataclass, field


@dataclass(init=True, repr=True)
class EpisodeInfo(object):
    """The EpisodeInfo class. Wraps the result from
    on_training_episode

    """

    episode_index: int = -1
    episode_reward: float = 0.0
    episode_iterations: int = 0
    total_execution_time: float = 0.0
    info: dict = field(default_factory=dict)

