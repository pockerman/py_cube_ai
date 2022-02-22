"""
Qlearning on CartPole-v0 using state aggregation
"""
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.td.q_learning import QLearning
from src.worlds.state_aggregation_cart_pole_env import StateAggregationCartPoleEnv
from src.algorithms.td.td_algorithm_base import TDAlgoConfig
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption


def plot_running_avg(avg_rewards):

    running_avg = np.empty(avg_rewards.shape[0])
    for t in range(avg_rewards.shape[0]):
        running_avg[t] = np.mean(avg_rewards[max(0, t-100) : (t+1)])
    plt.plot(running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Reward")
    plt.title("Running average")
    plt.show()


if __name__ == '__main__':
    GAMMA = 1.0
    ALPHA = 0.1
    EPS = 1.0

    env = StateAggregationCartPoleEnv(n_states=10, n_actions=2)
    q_algo_in = TDAlgoConfig()
    q_algo_in.train_env = env
    q_algo_in.alpha = ALPHA
    q_algo_in.gamma = GAMMA
    q_algo_in.n_episodes = 50000
    q_algo_in.n_itrs_per_episode = 10000
    q_algo_in.output_freq = 5000
    q_algo_in.policy = EpsilonGreedyPolicy(env=env, eps=EPS,
                                           decay_op=EpsilonDecayOption.INVERSE_STEP,
                                           min_eps=0.001)

    agent = QLearning(algo_in=q_algo_in)
    agent.train()

    plot_running_avg(agent.total_rewards)

