import gym
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from algorithms.reinforce import Reinforce



if __name__ == '__main__':

    GAMMA = 0.99
    LEARNING_RATE = 0.01
    EPISODES_TO_TRAIN = 4
    env = gym.make("CartPole-v0")
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    reinforce = Reinforce()
