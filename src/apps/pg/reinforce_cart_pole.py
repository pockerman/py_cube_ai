"""
Use REINFORCE to solve the CartPole environment
"""

from typing import Any
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from src.policies.policy_base import PolicyTorchBase
from src.algorithms.pg.reinforce import Reinforce

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(PolicyTorchBase):
    def __init__(self, env: Any, s_size=4, h_size=16, a_size=2) -> None:
        super(Policy, self).__init__(env=env)
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    policy = Policy(env=env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    reinforce = Reinforce(env=env, n_max_iterations=1000, gamma=0.1,
                          optimizer=optimizer, tolerance=1.0e-2, max_itrs_per_episode=100,
                          print_frequency=100, policy=policy)

    reinforce.train()


