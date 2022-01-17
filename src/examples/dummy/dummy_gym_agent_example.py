import gym
import matplotlib.pyplot as plt
from src.algorithms.dummy.dummy_gym_agent import DummyGymAgent
from src.algorithms.algo_config import AlgoConfig


if __name__ == '__main__':

    env = gym.make("MountainCar-v0")

    algo_in = AlgoConfig()
    algo_in.n_episodes = 1000
    algo_in.train_env = gym.make("MountainCar-v0")
    algo_in.render_env = True
    algo_in.render_env_freq = 10

    agent = DummyGymAgent(algo_in=algo_in, n_itrs_per_episode=100)
    agent.train()

    plt.plot(agent.rewards)
    plt.show()
