"""
Qlearning on CartPole-v0 using state aggregation
"""
import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.td.q_learning import QLearning
from src.worlds.state_aggregation_cart_pole_env import StateAggregationCartPoleEnv
from src.algorithms.td.td_algorithm_base import TDAlgoConfig
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from src.worlds.world_helpers import n_actions
from src.trainers.rl_serial_agent_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer


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

    agent_config = TDAlgoConfig(gamma=GAMMA, alpha=ALPHA,
                                n_itrs_per_episode=50000,
                                n_episodes=10000,
                                policy=EpsilonGreedyPolicy(n_actions=n_actions(env),
                                                           eps=EPS, decay_op=EpsilonDecayOption.INVERSE_STEP))

    agent = QLearning(agent_config)

    trainer_config = RLSerialTrainerConfig(n_episodes=50000, output_msg_frequency=5000)
    trainer = RLSerialAgentTrainer(trainer_config, agent=agent)
    trainer.train(env)

    plot_running_avg(agent.total_rewards)

