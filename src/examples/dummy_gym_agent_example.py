import gym
import matplotlib.pyplot as plt
from src.algorithms.dummy_gym_agent import DummyGymAgent


if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    num_games = 1000

    agent = DummyGymAgent(env=env, n_episodes=1000, tolerance=1.0e-4, n_itrs_per_episode=100)
    agent.train()

    plt.plot(agent.rewards)
    plt.show()
