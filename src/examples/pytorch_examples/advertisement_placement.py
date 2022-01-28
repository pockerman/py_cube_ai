"""
The following example demonstrates how to solve the
context-bandit problem. The example is taken from the book
Deep Reinforcement Learning in Action
This leads us to state spaces. The n-armed bandit problem we started with
had an n-element action space (the space or set of all possible actions),
but there was no concept of state. That is,
there was no information in the environment that would help us choose a good arm.
The only way we could figure out which arms were good is by trial and error.
In the ad problem, we know the user is buying something on a particular site, which may give
us some information about that user’s preferences and could help guide our
decision about which ad to place. We call this contextual information a
state and this new class of problems contextual bandits

 we’ll be using PyTorch to build the neural network. In this case, we’re going to build a two-layer feedforward neural
 network that uses rectified linear units (ReLU) as the activation function.
 The first layer accepts a 10-element one-hot (also known as 1-of-K, where all elements but one are 0)
 encoded vector of the state, and the final layer returns a 10-element
 vector representing the predicted reward for each action given the state.

"""

import numpy as np
import random
import torch

from src.utils.array_utils import build_one_hot_encoding
from src.policies.softmax_policy import SoftMaxPolicy

N_ACTIONS = 10
TAU = 1.2

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

    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms, arms)

    def reward(self, prob):
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

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward


def train(env, epochs=5000, learning_rate=1e-2):
    """
    Once we enter the main training for loop, we’ll run our neural network model with the randomly
    initialized current state vector. It will return a vector that represents its
    guess for the values of each of the possible actions. At first, the model will
    output a bunch of random values since it is not trained.
    We’ll run the softmax function over the model’s output to generate a probability
    distribution over the actions. We’ll then select an action using the
    environment’s choose_arm(...) function, which will return the reward
    generated for taking that action; it will also update the environment’s current state.
    We’ll turn the reward (which is a non-negative integer) into a one-hot vector that
    we can use as our training data. We’ll then run one step of backpropagation with this reward vector,
     given the state we gave the model. Since we’re using a neural network model as our
     action-value function, we no longer have any sort of action-value array storing “memories;”
    everything is being encoded in the neural network’s weight parameters. The whole train function is shown in the following listing.
    :param env:
    :param epochs:
    :param learning_rate:
    :return:
    """
    cur_state = torch.Tensor(build_one_hot_encoding(arms, env.get_state()))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rewards = []
    softmax = SoftMaxPolicy(n_actions=N_ACTIONS, tau=TAU)

    for i in range(epochs):
        y_pred = model(cur_state)
        av_softmax = softmax(y_pred.data.numpy(), tau=2.0)
        av_softmax /= av_softmax.sum()
        choice = np.random.choice(arms, p=av_softmax)
        cur_reward = env.choose_arm(choice)
        one_hot_reward = y_pred.data.numpy().copy()
        one_hot_reward[choice] = cur_reward
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(build_one_hot_encoding(arms,env.get_state()))
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
    train(env=env)

