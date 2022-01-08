

import gym
import highway_env
import numpy as np
import os
import collections
from collections import defaultdict, deque


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
#import matplotlib.image as mpimg
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.animation as animation
#import matplotlib

from src.utils import INFO
from src.algorithms.td.q_learning import QLearning
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption

# We have to limit the states to small finite number.
N_STATES = 36
GAMMA = 1.0
INIT_LR = 1.0
MIN_LR = 0.003
EPS = 0.02


def get_pos_vel_bins(env, obs, n_states):
    """
    Returns the bin index for position and velocity
    for the given observation
    :param env:
    :param obs:
    :param n_states:
    :return:
    """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    pos = int((obs[0] - env_low[0])/env_dx[0]) # position value
    vel = int((obs[1] - env_low[1])/env_dx[1]) # velocity value
    return pos, vel


class MyMountainCarEnv(object):

    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.env.seed(42)

    def reset(self) -> tuple:
        result = self.env.reset()
        return get_pos_vel_bins(env=env, obs=result, n_states=N_STATES)

    def step(self, action):
        next_state, reward, done, info = self.env.on_episode(action=action)
        return get_pos_vel_bins(env=env, obs=next_state, n_states=N_STATES), reward, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()

    def render(self, mode="human", **kwargs):
        return self.env.render(mode=mode, **kwargs)


class MyQLearning(QLearning):
    def __init__(self, n_episodes, tolerance,
                          env, gamma, policy, n_itrs_per_episode) -> None:
        super(MyQLearning, self).__init__(n_episodes=n_episodes, tolerance=tolerance,
                                          env=env, gamma=gamma, policy=policy,
                                          n_itrs_per_episode=n_itrs_per_episode, alpha=INIT_LR)

    def actions_before_training_begins(self, **options) -> None:

        super(MyQLearning, self).actions_before_training_begins(**options)

        # initialize Q(s, a) as a table of shape (N_STATES, N_STATES, 3)
        self._q = np.zeros((N_STATES, N_STATES, 3))

    def actions_before_episode_begins(self, **options):
        super(MyQLearning, self).actions_before_episode_begins(**options)

        self.alpha = max(MIN_LR, INIT_LR * (0.85 ** (self.current_episode_index // 100)))


if __name__ == '__main__':

    env = MyMountainCarEnv()

    # Only 3 actions allowed move left(0), not move(1) and move right(2).
    print('{0} Action Space for Mountain Car Env: {1}'.format(INFO, str(env.action_space)))

    # From observation space we get position and speed of the agent.
    print('{0} Observation Space for Mountain Car Env: {1}'.format(INFO, str(env.observation_space)))

    print('{0} Observation space and speed values (MAX): {1}'.format(INFO, str(env.observation_space.high)))
    print('{0} Observation space and speed values (MIN): {1}'.format(INFO, str(env.observation_space.low)))

    policy_init = EpsilonGreedyPolicy(env=env, decay_op=EpsilonDecreaseOption.NONE, eps=0.02)

    qlearner = MyQLearning(n_episodes=10000, tolerance=1e-10,
                           env=env, gamma=GAMMA, policy=policy_init,
                           n_itrs_per_episode=10000)

    qlearner.train()




