import gym

from src.algorithms.planning.monte_carlo_tree_search import MCTreeSearch
from src.algorithms.planning.monte_carlo_tree_search import MCTreeSearchInput

if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    input = MCTreeSearchInput()
    input.train_env = env
    input.render_env = True
    input.n_episodes = 5000



