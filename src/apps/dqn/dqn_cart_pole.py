"""
Cart-pole problem using DQN. The implementation
is taken from the  PyTorch documentation here
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gym
import torch
import torch.optim as optim
from src.algorithms import DQN
from src.networks.nn_base import NNBase


class TargetNet(NNBase):
    """
    The target network
    """
    pass


class QNet(NNBase):
    """
    Q-network
    """
    pass


if __name__ == '__main__':

    # various constants
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    NUM_EPISODES = 50
    SEED = 42
    BUFFER_SIZE = 10000
    ENV_NAME = 'CartPole-v0'

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    print("Start training DQN on {}".format(ENV_NAME))


    # The environmen
    env = gym.make(ENV_NAME).unwrapped

    target_net = TargetNet()
    policy_net = QNet()

    optimizer = optim.RMSprop(policy_net.parameters())
    agent = DQN(env=env, target_network=target_net, local_net=policy_net,
                 n_max_iterations=NUM_EPISODES, tolerance=1.0e-8, update_frequency=TARGET_UPDATE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, tau=0.4,
                 steps_per_iteration=100, state_size=10, action_size=env.action_space.n,
                 eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=device,
                 buffer_size=BUFFER_SIZE, seed=SEED)


    # Train the agent
    agent.train()

    print("Finished training DQN on {}".format(ENV_NAME))