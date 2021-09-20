import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from algorithms.dqn.dqn import DQN
from networks.nn_base import NNBase


class DQNNet(NNBase):
    def __init__(self, seed: int, state_size: int, action_size:int,
                 fc1_units: int=64, fc2_units: int=64) -> None:
        """Initialize parameters and build model.

        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """

        super(DQNNet, self).__init__()
        self._seed = torch.manual_seed(seed)
        self._fc1 = nn.Linear(state_size, fc1_units)
        self._fc2 = nn.Linear(fc1_units, fc2_units)
        self._fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self._fc1(state))
        x = F.relu(self._fc2(x))
        return self._fc3(x)


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    env.seed(0)
    device = 'cpu'

    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    state_size = 8
    action_size = 4
    seed = 0

    target_network = DQNNet(seed=seed, state_size=state_size, action_size=action_size).to(device)
    local_net = DQNNet(seed=seed, state_size=state_size, action_size=action_size).to(device)
    optimizer = optim.Adam(local_net.parameters(), lr=5e-4)
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e5)
    batch_size = 64

    agent = DQN(env=env, target_network=target_network, local_net=local_net,
                n_max_iterations=2000, tolerance=1.0e-4, update_frequency=4,
                steps_per_iteration=1000, batch_size=batch_size, gamma=gamma,
                state_size=state_size, action_size=action_size,
                optimizer=optimizer, tau=tau, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                buffer_size=buffer_size)

    train_result = agent.train()
    print(train_result)

    scores = agent.scores

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

