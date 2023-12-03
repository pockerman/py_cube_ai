"""
Use REINFORCE to solve the CartPole environment
"""

import numpy as np
from typing import Any, List
import matplotlib.pyplot as plt
import gym


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


from pycubeai.algorithms.pg.reinforce import Reinforce, ReinforceConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y


class CartPoleNetwork(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2) -> None:
        super(CartPoleNetwork, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def on_state(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return m.log_prob(action)


def loss_fn(preds, r) -> torch.Tensor:
    """
    The loss function expects an array of action probabilities
    for the actions that were taken and the discounted rewards
    It computes the log of the probabilities, multiplies
    by the discounted rewards, sums them all and flips the sign
    :param preds:
    :param r:
    :return:
    """
    # return -1 * torch.sum(r * torch.log(preds))

    loss = []
    for log_prob in preds:
        loss.append(-log_prob * r)
    #loss = torch.cat(loss).sum()
    loss = torch.stack(loss, dim=0).sum()
    return loss


def action_selector(probs: List) -> int:
    m = Categorical(probs)
    action = m.sample()
    return action.item()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    model = CartPoleNetwork()

    # algorithm configuration
    reinforce_config = ReinforceConfig()
    reinforce_config.optimizer = optim.Adam(model.parameters(), lr=0.0009)
    reinforce_config.action_selector = action_selector
    reinforce_config.train_env = env
    reinforce_config.policy_network = model
    reinforce_config.loss_func = loss_fn
    reinforce_config.n_episodes = 500
    reinforce_config.n_itrs_per_episode = 200
    reinforce_config.gamma = 0.99
    reinforce_config.mean_reward_for_exit = -1.0 #195.0
    reinforce_config.output_freq = 20

    # the agent to train
    reinforce_agent = Reinforce(algo_in=reinforce_config)
    reinforce_agent.train()

    score = np.array(reinforce_agent.iterations_per_episode)
    avg_score = running_mean(score, 50)

    plt.figure(figsize=(10, 7))
    plt.ylabel("Episode Duration", fontsize=22)
    plt.xlabel("Training Epochs", fontsize=22)
    plt.plot(avg_score, color='green')
    plt.show()

    reinforce_agent.play(env=env, n_games=100, max_duration_per_game=200)


