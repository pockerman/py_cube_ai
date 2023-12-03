"""Episodic semi-gradient SARSA algorithm applied
on MountainCar-v0.

The initial version of the algorithm is
taken from
https://livevideo.manning.com/module/56_8_8/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/episodic-semi-gradient-control%3a-sarsa?

"""

import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeVar

from pycubeai.algorithms.td.episodic_sarsa_semi_gradient import EpisodicSarsaSemiGrad, SemiGradSARSAConfig
from pycubeai.worlds.state_aggregation_world_wrapper import StateAggregationEnvWrapper
from pycubeai.worlds.state_aggregation_mountain_car_env import StateAggregationMountainCarEnv
from pycubeai.trainers.rl_serial_algorithm_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer
from pycubeai.utils import INFO

GAMMA = 1
NUM_EPISODES = 500
NUM_RUNS = 50
N_BINS = 8
N_LAYERS = 8
N_TILES = 8

State = TypeVar('State')
Action = TypeVar('Action')
TiledState = TypeVar('TiledState')


class TiledMountainCarEnv(StateAggregationEnvWrapper):

    def __init__(self):
        super(TiledMountainCarEnv, self).__init__(env=gym.make('MountainCar-v0'), n_actions=3, n_states=8 * 8 * 8)
        self.bins = self.create_bins()
        self.raw_env._max_episode_steps = 1000

    def create_bins(self) -> list:
        """
        Create the bins that the state variables of
        the underlying environment will be distributed
        :return: A list of bins for every state variable
        """

        pos_tile_width = (0.5 + 1.2) / N_BINS * 0.5
        vel_tile_width = (0.07 + 0.07) / N_BINS * 0.5
        pos_bins = np.zeros((N_LAYERS, N_BINS))
        vel_bins = np.zeros((N_LAYERS, N_BINS))

        for i in range(N_LAYERS):
            pos_bins[i] = np.linspace(-1.2 + i * pos_tile_width, 0.5 + i * pos_tile_width / 2, N_BINS)
            vel_bins[i] = np.linspace(-0.07 + 3 * i * vel_tile_width, 0.07 + 3 * i * vel_tile_width / 2, N_BINS)

        return [pos_bins, vel_bins]

    def get_tiled_state(self, obs: State, action: Action) -> TiledState:
        """
        Returns the tiled states for the given observation
        :param obs: The observation to be tiled
        :param action: The action corresponding to the states
        :return: TiledState
        """

        position, velocity = obs

        tiled_state = np.zeros(N_TILES * N_TILES * N_TILES * self.n_actions)
        for row in range(N_LAYERS):
            if self.bins[0][row][0] < position < self.bins[0][row][N_TILES - 1]:
                if self.bins[1][row][0] < velocity < self.bins[1][row][N_TILES - 1]:
                    x = np.digitize(position, self.bins[0][row])
                    y = np.digitize(velocity, self.bins[1][row])
                    idx = (x + 1) * (y + 1) + row * N_TILES ** 2 - 1 + action * N_LAYERS * N_TILES ** 2
                    tiled_state[idx] = 1.0
                else:
                    break
            else:
                break
        return tiled_state


class Policy(object):

    def __init__(self, epsilon: float):
        self.eps = epsilon

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

    env = StateAggregationMountainCarEnv(version="v0", n_states=8 * 8 * 8)

    lrs = [0.01, 0.1, 0.2]
    episode_lengths = np.zeros((3, NUM_EPISODES, NUM_RUNS))

    x = [i for i in range(episode_lengths.shape[1])]

    for k, lr in enumerate(lrs):

        print("==================================")
        print("{0} Working with learning rate {1}".format(INFO, lr))
        print("==================================")
        # for each learning rate we do a certain number
        # of runs
        for j in range(NUM_RUNS):
            print("{0}: run {1}".format(INFO, j))
            policy = Policy(epsilon=1.0)

            agent_config = SemiGradSARSAConfig(n_episodes=NUM_EPISODES,
                                               n_itrs_per_episode=2000, policy=policy, alpha=lr,
                                               gamma=GAMMA, dt_update_frequency=100, dt_update_factor=1.0)

            agent = EpisodicSarsaSemiGrad(algo_config=agent_config)

            trainer_config = RLSerialTrainerConfig(n_episodes=NUM_EPISODES, tolerance=1.0e-4, output_msg_frequency=100)
            trainer = RLSerialAgentTrainer(trainer_config, agent)
            trainer.train(env)

            counters = agent.counters

            for item in counters:
                episode_lengths[k][item-1][j] = counters[item]
        print("==================================")
        print("==================================")

    averaged1 = np.mean(episode_lengths[0], axis=1)
    averaged2 = np.mean(episode_lengths[1], axis=1)
    averaged3 = np.mean(episode_lengths[2], axis=1)

    plt.plot(averaged1, 'r--')
    plt.plot(averaged2, 'b--')
    plt.plot(averaged3, 'g--')

    plt.legend(('alpha = 0.01', 'alpha = 0.1', 'alpha = 0.2'))
    plt.title("Episode semi-gradient SARSA (MountainCar-v0)")
    plt.xlabel("Episode")
    plt.xlabel("Number of iterations")
    plt.show()
    env.close()
