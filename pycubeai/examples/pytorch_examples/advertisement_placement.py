"""
The following example demonstrates how to solve the
context-bandit problem. The example is taken from the book
Deep Reinforcement Learning in Action by Manning.
The book GitHub repository is at https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction

The n-armed bandit problem has an n-element action space meaning the space or set of all possible actions.
However, there is no concept of state.
This means  that there is no information in the environment that would help us choose a good arm.
The only way we could figure out which arms were good is by trial and error.

In the ad problem, we know the user is buying something on a particular site, which may give
us some information about that user’s preferences and could help guide our
decision about which ad to place. We call this contextual information
state and this new class of problems contextual bandits

This example is using PyTorch to build a neural network that represents the  state-action value function.
In particular, we are going to build a two-layer feedforward neural
network that uses rectified linear units (ReLU) as the activation function.
The first layer accepts a 10-element one-hot encoded vector of the state,
and the final layer returns a 10-element
vector representing the predicted reward for each action given the state.

"""

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

from pycubeai.utils.array_utils import build_one_hot_encoding
from pycubeai.policies.softmax_policy import SoftMaxPolicy
from pycubeai.utils import INFO

N_ACTIONS = 10
TAU = 2.0

def running_mean(x, N=50):
    """
    Helper function for plotting running mean
    :param x:
    :param N:
    :return:
    """
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


class ContextBandit:
    """
    The class that represents the environment
    """
    def __init__(self, arms=10):
        self.arms = arms
        self.bandit_matrix = None
        self.state = None
        self.init_distribution(arms)
        self.update_state()

    def step(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward

    def init_distribution(self, arms: int):
        self.bandit_matrix = np.random.rand(arms, arms)

    def reward(self, prob: float):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        """
        Returns a state sampled randomly from a uniform distribution.
        See the update_state function
        :return:
        """
        return self.state

    def update_state(self):
        self.state = np.random.randint(0, self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])


def train(env, network, epochs=5000, learning_rate=1e-2):
    """
    Main training function.

    We initialize the current state randomly and transform into a one-hot-encoded tensor.
    In the training loop, we evaluate the model
    Once we enter the main training for loop, we’ll run our neural network model with the randomly
    initialized current state vector. It will return a vector that represents its
    guess for the values of each of the possible actions.

    At first, the model will output a bunch of random values since it is not trained.
    We’ll run the softmax function over the model’s output to generate a probability
    distribution over the actions. We’ll then select an action using the
    environment’s step(...) function, which will return the reward
    generated for taking that action; it will also update the environment’s current state.

    We turn the reward (which is a non-negative integer) into a one-hot vector that
    can be used as the training data. After that we run one step of backpropagation using this reward vector,
    for the state we gave the model. Since we’re using a neural network model as our
    action-value function, we no longer have any sort of action-value array storing “memories;”
    everything is being encoded in the neural network’s weight parameters.
    
    """
    cur_state = torch.Tensor(build_one_hot_encoding(n=arms, pos=env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    softmax = SoftMaxPolicy(n_actions=N_ACTIONS, tau=TAU)

    for i in range(epochs):

        # get the predictions from the model
        y_pred = network(cur_state)

        # make a choice
        av_softmax = softmax.softmax_values(y_pred.data.numpy())
        av_softmax /= av_softmax.sum()
        choice = np.random.choice(arms, p=av_softmax)

        # step in the environment
        cur_reward = env.step(choice)

        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)

        print("{0} On Episode={1} Loss={2}".format(INFO, i, torch.mean((y_pred - reward)**2)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(build_one_hot_encoding(arms, env.get_state()))
    return np.array(rewards)


if __name__ == '__main__':

    arms = 10
    N, D_in, H, D_out = 1, arms, 100, arms

    # create a PyTorch model
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.ReLU(),
    )

    # loss function
    loss_fn = torch.nn.MSELoss()

    # set up the environment
    env = ContextBandit(arms)

    # train the model
    rewards = train(env=env, network=model)

    plt.plot(running_mean(rewards, N=500))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
