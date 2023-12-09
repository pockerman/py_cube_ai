"""
Solves the armed-bandit problem using epsilon greedy policy.
The code in this example, is edited from the book
Deep Reinforcement Learning in Action by Manning publications
"""
import matplotlib.pyplot as plt
import numpy as np
import random

from pycubeai.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption


def get_reward(prob, n=10):
    """
    Returns the reward for the given probability
    :param prob:
    :param n:
    :return:
    """
    reward = 0
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward


def update_record(record, action, r):
    
    new_r = (record[action,0] * record[action, 1] + r) / (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_r
    return record


if __name__ == '__main__':

    n = 10
    policy = EpsilonGreedyPolicy(n_actions=n, eps=0.2, decay_op=EpsilonDecayOption.NONE)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Average Reward")
    fig.set_size_inches(9, 5)
    rewards = [0]
    probs = np.random.rand(n)

    #  10 actions x 2 columns
    # # Columns: Count #, Avg Reward
    record = np.zeros(shape=(n, 2))

    for i in range(500):
        values = record[:, 1]
        action = policy.choose_action_index(values=values)
        r = get_reward(probs[action])
        record = update_record(record, action, r)
        mean_reward = ((i + 1) * rewards[-1] + r) / (i + 2)
        rewards.append(mean_reward)
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()

