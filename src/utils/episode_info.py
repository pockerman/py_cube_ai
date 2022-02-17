"""
Helper class to wrap return result from
training on an episode
"""


class EpisodeInfo(object):
    def __init__(self):
        self.episode_reward: float = 0.0
        self.episode_iterations: int = 0
        self.info = {}