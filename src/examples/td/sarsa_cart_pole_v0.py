"""
SARSA algorithm on CartPole-v0 environment. Since the environment
has a continuous state vector we perform state aggregation
"""
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.td.td_algorithm_base import TDAlgoInput
from src.algorithms.td.sarsa import Sarsa
from src.worlds.state_aggregation_cart_pole_env import StateAggregationCartPoleEnv
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption


def plot_running_avg(avg_rewards):

    running_avg = np.empty(avg_rewards.shape[0])
    for t in range(avg_rewards.shape[0]):
        running_avg[t] = np.mean(avg_rewards[max(0, t-100) : (t+1)])
    plt.plot(running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Average")
    plt.title("Running average")
    plt.show()

if __name__ == '__main__':

    GAMMA = 1.0
    ALPHA = 0.1
    EPS = 1.0
    env = StateAggregationCartPoleEnv(n_states=10, n_actions=2, state_var_idx=4)

    sarsa_in = TDAlgoInput()
    sarsa_in.gamma = GAMMA
    sarsa_in.alpha = ALPHA
    sarsa_in.train_env = env
    sarsa_in.policy = EpsilonGreedyPolicy(env=env, eps=EPS,
                                          decay_op=EpsilonDecayOption.INVERSE_STEP,
                                          min_eps=0.001)
    sarsa_in.n_episodes = 50000
    sarsa_in.output_freq = 5000

    sarsa = Sarsa(algo_in=sarsa_in)
    sarsa.actions_before_training_begins()

    sarsa.train()

    # plot average reward per episode
    avg_reward = sarsa.avg_rewards

    plt.plot(avg_reward)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Average reward per episode")
    plt.show()

    plot_running_avg(avg_rewards=sarsa.total_rewards)
