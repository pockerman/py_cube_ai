"""
Train a Q-learning agent on Frozen Lake
"""
import gym

from pycubeai.algorithms.td.q_learning import QLearning, TDAlgoConfig
from pycubeai.trainers.rl_serial_algorithm_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer
from pycubeai.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from pycubeai.worlds.world_helpers import n_actions


if __name__ == '__main__':

    N_EPISODES = 1000
    N_ITRS_EPISODES = 200
    GAMMA = 0.9
    ALPHA = 0.2
    ENV_NAME = "FrozenLake-v0"

    test_env = gym.make(ENV_NAME)

    agent_config = TDAlgoConfig(gamma=0.9, alpha=ALPHA,
                                n_itrs_per_episode=N_EPISODES,
                                n_episodes=N_ITRS_EPISODES,
                                policy=EpsilonGreedyPolicy(n_actions=n_actions(test_env),
                                                           eps=1.0, decay_op=EpsilonDecayOption.INVERSE_STEP))
    agent = QLearning(agent_config)

    trainer_config = RLSerialTrainerConfig(n_episodes=N_EPISODES, output_msg_frequency=10)
    trainer = RLSerialAgentTrainer(config=trainer_config, algorithm=agent)
    ctrl_res = trainer.train(test_env)

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")


