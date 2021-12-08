import gym
import numpy as np
from src.algorithms.algorithm_base import AlgorithmBase


class DummyGymAgent(AlgorithmBase):
    """
    The DummyGymAgent class. Dummy class to play with OpenAI-Gym environments
    """

    def __init__(self, n_episodes: int, n_itrs_per_episode: int,
                 tolerance: float, env: gym.Env) -> None:
        super(DummyGymAgent, self).__init__(n_episodes=n_episodes,
                                            tolerance=tolerance, env=env)
        self.n_itrs_per_episode = n_itrs_per_episode
        self.rewards = np.zeros(self.n_episodes)

    def step(self, **options) -> None:
        """
        Do one step of the algorithm
        """
        done = False
        episode_rewards = 0

        while not done:
            action = self.train_env.action_space.sample()
            observation, reward, done, info = self.train_env.step(action=action)
            episode_rewards += reward

        self.rewards[self.current_episode_index] = episode_rewards

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

