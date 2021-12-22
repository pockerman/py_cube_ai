import gym
import numpy as np
from src.algorithms.algorithm_base import AlgorithmBase
from src.algorithms.algo_input import AlgoInput


class DummyGymAgent(AlgorithmBase):
    """
    The DummyGymAgent class. Dummy class to play with OpenAI-Gym environments
    """

    def __init__(self, algo_in: AlgoInput, n_itrs_per_episode: int) -> None:
        super(DummyGymAgent, self).__init__(algo_in=algo_in)
        self.n_itrs_per_episode = n_itrs_per_episode
        self.rewards = np.zeros(self.n_episodes)

    def step(self, **options) -> None:
        """
        Do one step of the algorithm
        """
        done = False
        episode_rewards = 0

        for episode_itr in range(self.n_itrs_per_episode):
            action = self.train_env.action_space.sample()
            observation, reward, done, info = self.train_env.step(action=action)
            episode_rewards += reward

            if self.render_env and episode_itr % self.render_env_freq == 0:
                self.train_env.render(mode="human")

            if done:
                break

        self.rewards[self.current_episode_index] = episode_rewards

    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

