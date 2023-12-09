import gym
import numpy as np
from typing import TypeVar
import matplotlib.pyplot as plt

from pycubeai.utils import INFO
from pycubeai.algorithms.td.q_learning import QLearning
from pycubeai.algorithms import TDAlgoConfig
from pycubeai.trainers import RLSerialAgentTrainer, RLSerialTrainerConfig
from pycubeai.worlds import StateAggregationMountainCarEnv
from pycubeai.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption


Env = TypeVar('Env')

# We have to limit the states to small finite number.
N_STATES = 36
GAMMA = 1.0
INIT_LR = 1.0
MIN_LR = 0.003
EPS = 0.02
N_EPISODES = 10000


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
    def __init__(self, algo_config: TDAlgoConfig) -> None:
        super(MyQLearning, self).__init__(algo_config=algo_config)

    def actions_before_training_begins(self, **options) -> None:

        super(MyQLearning, self).actions_before_training_begins(**options)

        # initialize properly the state
        for state in env.state_space:
            for action in range(env.n_actions):
                self.q_table[state, action] = 0.0

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options):
        super(MyQLearning, self).actions_before_episode_begins(**options)

        self.config.alpha = max(MIN_LR, INIT_LR * (0.85 ** (episode_idx // 100)))


if __name__ == '__main__':

    #env = MyMountainCarEnv()

    env = StateAggregationMountainCarEnv(version="v0", n_states=N_STATES)

    # Only 3 actions allowed move left(0), not move(1) and move right(2).
    print('{0} Action Space for Mountain Car Env: {1}'.format(INFO, str(env.action_space)))

    print("{0} Car position bounds {1}".format(INFO, env.car_position_bounds))
    print("{0} Car velocity bounds {1}".format(INFO, env.car_velocity_bounds))

    algo_config = TDAlgoConfig(n_itrs_per_episode=N_EPISODES,
                               policy=EpsilonGreedyPolicy(n_actions=env.n_actions,
                                                          decay_op=EpsilonDecayOption.NONE, eps=EPS),
                               n_episodes=10000, gamma=GAMMA, alpha=INIT_LR)

    qlearner = MyQLearning(algo_config=algo_config)

    trainer_config = RLSerialTrainerConfig(n_episodes=N_EPISODES, output_msg_frequency=100)
    trainer = RLSerialAgentTrainer(config=trainer_config, algorithm=qlearner)

    trainer.train(env)




