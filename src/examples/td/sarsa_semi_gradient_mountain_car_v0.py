"""
Episodic semi-gradient SARSA algorithm applied
on MountainCar-v0.

The initial version of the algorithm is
taken from
https://livevideo.manning.com/module/56_8_8/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/episodic-semi-gradient-control%3a-sarsa?
"""

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.td.episodic_sarsa_semi_gradient import EpisodicSarsaSemiGrad

GAMMA = 1
NUM_EPISODES = 500
NUM_RUNS = 50


def get_bins(n_bins: int=8, n_layers: int=8):

    pos_tile_width = (0.5 + 1.2) / n_bins * 0.5
    vel_tile_width = (0.07 + 0.07) / n_bins * 0.5
    pos_bins = np.zeros((n_layers, n_bins))
    vel_bins = np.zeros((n_layers, n_bins))

    for i in range(n_layers):
        pos_bins[i] = np.linspace(-1.2 + i * pos_tile_width, 0.5 + i * pos_tile_width/2, n_bins)
        vel_bins[i] = np.linspace(-0.07 + 3* i * vel_tile_width, 0.07 + 3 * i * vel_tile_width / 2, n_bins)

    return pos_bins, vel_bins


def tile_state(pos_bins, vel_bins, action, obs, n_tiles: int = 8, n_layers: int = 8, n_actions=3):
    position, velocity = obs

    tiled_state = np.zeros(n_tiles * n_tiles * n_tiles * n_actions)
    for row in range(n_layers):
        if pos_bins[row][0] < position < pos_bins[row][n_tiles - 1]:
            if vel_bins[row][0] < velocity < vel_bins[row][n_tiles - 1]:
                x = np.digitize(position, pos_bins[row])
                y = np.digitize(velocity, vel_bins[row])
                idx = (x + 1) * (y + 1) + row * n_tiles**2 - 1 + action * n_layers * n_tiles**2
                tiled_state[idx] = 1.0
            else:
                break
        else:
            break
    return tiled_state


class TiledMountainCarEnv(gym.Env):

    def __init__(self):
        super(TiledMountainCarEnv, self).__init__()
        self._env = gym.make('MountainCar-v0')
        self._env._max_episode_steps = 1000
        self.pos_bins, self.vel_bins = get_bins()

    @property
    def n_states(self):
        return 8 * 8 * 8

    @property
    def n_actions(self):
        return 3

    @property
    def max_episode_steps(self):
        return self._env._max_episode_steps

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action=action)

    def close(self):
        self._env.close()

    def get_state(self, obs, action):
        return tile_state(pos_bins=self.pos_bins, vel_bins=self.vel_bins,
                          obs=obs, action=action)


class Policy(object):

    def __init__(self, epsilon: float, env: TiledMountainCarEnv):
        self.eps = epsilon
        self.env = env
        self.pos_bins, self.vel_bins = get_bins()

    def __call__(self, values, observation, **kwargs):

        # select greedy action with probability epsilon
        if random.random() < 1.0 - self.eps:
            best = np.argmax(values)
            return best
        else:
            # otherwise, select an action randomly
            return random.choice([0, 1, 2])

    def actions_after_episode(self, episode_idx: int, **options) -> None:

        if self.eps -2 / episode_idx > 0:
            self.eps -= 2 / episode_idx
        else:
            self.eps = 0.0


if __name__ == '__main__':

    env = TiledMountainCarEnv()

    lrs = [0.01, 0.1, 0.2]
    episode_lengths = np.zeros((3, NUM_EPISODES, NUM_RUNS))

    x = [i for i in range(episode_lengths.shape[1])]

    for k, lr in enumerate(lrs):

        # for each learning rate we do a certain number
        # of runs
        for j in range(NUM_RUNS):
            policy = Policy(epsilon=1.0, env=env)
            agent = EpisodicSarsaSemiGrad(env=env, tolerance=1.0e-4, gamma=GAMMA, alpha=lr,
                                          n_episodes=NUM_EPISODES, n_itrs_per_episode=2000, policy=policy)
            agent.train()

            counters = agent.counters

            for item in counters:
                episode_lengths[k][item-1][j] = counters[item]

    averaged1 = np.mean(episode_lengths[0], axis=1)
    averaged2 = np.mean(episode_lengths[1], axis=1)
    averaged3 = np.mean(episode_lengths[2], axis=1)

    plt.plot(averaged1, 'r--')
    plt.plot(averaged1, 'b--')
    plt.plot(averaged1, 'g--')

    plt.legend(('alpha = 0.01', 'alpha = 0.1', 'alpha = 0.2'))
    plt.show()
