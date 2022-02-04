from src.algorithms.dqn.dqn_target_network_grid_world import DQNTargetNetworkGridWorld, DQNGridWorldConfig
from src.worlds.grid_world import Gridworld, GridworldInitMode
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption

EPS = 0.3
GAMMA = 0.9
if __name__ == '__main__':

    # configuration for the algorithm
    config = DQNGridWorldConfig()
    config.memory_size = 1000
    config.n_episodes = 5000
    config.batch_size = 200
    config.synchronize_frequency = 500
    config.policy = EpsilonGreedyPolicy(eps=EPS, decay_op=EpsilonDecreaseOption.NONE)
    config.gamma = GAMMA
    config.train_env = Gridworld(size=4, mode=GridworldInitMode.RANDOM, noise_factor=100)

    dqn_agent = DQNTargetNetworkGridWorld(config=config)
    dqn_agent.train()
