"""
Double Q-learning on CartPole-v0. The example has been edited
from Reinforcement Learning in Motion video series
"""

import matplotlib.pyplot as plt
import numpy as np


from pycubeai.algorithms.td.double_q_learning import DoubleQLearning
from pycubeai.worlds.state_aggregation_cart_pole_env import StateAggregationCartPoleEnv
from pycubeai.algorithms.td.td_algorithm_base import TDAlgoConfig
from pycubeai.policies.epsilon_greedy_policy import EpsilonDoubleGreedyPolicy, EpsilonDecayOption
from pycubeai.trainers.rl_serial_algorithm_trainer import RLSerialAgentTrainer, RLSerialTrainerConfig


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

    env = StateAggregationCartPoleEnv(n_states=10)

    q_algo_config = TDAlgoConfig(alpha=ALPHA, gamma=GAMMA, n_episodes=50000,
                                 n_itrs_per_episode=10000,
                                 policy=EpsilonDoubleGreedyPolicy(n_actions=env.n_actions, eps=EPS,
                                                                  decay_op=EpsilonDecayOption.NONE,
                                                                  min_eps=0.001))

    agent = DoubleQLearning(algo_config=q_algo_config)

    trainer_config = RLSerialTrainerConfig(n_episodes=50000, output_msg_frequency=5000)
    trainer = RLSerialAgentTrainer(algorithm=agent, config=trainer_config)

    trainer.train(env)

    plot_running_avg(agent.total_rewards)

