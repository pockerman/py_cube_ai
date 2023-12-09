import gym
import numpy as np
from pycubeai.algorithms.td.td_zero_semi_gradient import TDZeroSemiGrad


GAMMA = 1.0
NUM_EPISODES = 20000
N_ITRS_PER_EPISODE = 1000


class Policy(object):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs) -> int:

        print(args)
        if args[0] < 0:
            return 0
        return 2


def get_bins(n_bins: int=8, n_layers: int=8):

    pos_tile_width = (0.5 + 1.2) / n_bins * 0.5
    vel_tile_width = (0.07 + 0.07) / n_bins * 0.5
    pos_bins = np.zeros((n_layers, n_bins))
    vel_bins = np.zeros((n_layers, n_bins))

    for i in range(n_layers):
        pos_bins[i] = np.linspace(-1.2 + i * pos_tile_width, 0.5 + i * pos_tile_width/2, n_bins)
        vel_bins[i] = np.linspace(-1.2 + i * vel_tile_width, 0.5 + i * vel_tile_width / 2, n_bins)

    return pos_bins, vel_bins


def tile_state(pos_bins, vel_bins, obs, n_tiles: int = 8, n_layers: int = 8):
    position, velocity = obs

    tiled_state = np.zeros(n_tiles * n_tiles * n_tiles)
    for row in range(n_layers):
        if pos_bins[row][0] < position < pos_bins[row][n_tiles - 1]:
            if vel_bins[row][0] < velocity < vel_bins[row][n_tiles - 1]:
                x = np.digitize(position, pos_bins[row])
                y = np.digitize(velocity, vel_bins[row])
                idx = (x + 1) * (y + 1) + row * 4 - 1
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
        self.pos_bins, self.vel_bins = get_bins()

    def n_states(self):
        return 8 * 8 * 8

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.on_episode(action=action)

    def close(self):
        self._env.close()

    def get_state(self, obs):
        return tile_state(pos_bins=self.pos_bins, vel_bins=self.vel_bins, obs=obs)


class TDSemiGred(TDZeroSemiGrad):

    def __init__(self, env: TiledMountainCarEnv, alpha: float):
        super(TDSemiGred, self).__init__(n_episodes=NUM_EPISODES, tolerance=1.0e-4, env=env,
                                         gamma=GAMMA, alpha=alpha, n_itrs_per_episode=N_ITRS_PER_EPISODE,
                                         policy=Policy())

        self.near_exit = np.zeros(int(NUM_EPISODES / 1000))
        self.left_side = np.zeros(int(NUM_EPISODES / 1000))

    def actions_after_episode_ends(self, **options):

        if self.current_episode_index % 1000 == 0:
            tiled_state = tile_state(self.train_env.pos_bins,
                                     self.train_env.vel_bins, (0.43, 0.054))
            self.near_exit[self.current_episode_index // 1000] = self.get_state_value(tiled_state)

            tiled_state = tile_state(self.train_env.pos_bins,
                                     self.train_env.vel_bins, (-1.1, 0.001))
            self.left_side[self.current_episode_index // 1000] = self.get_state_value(tiled_state)

        if self.current_episode_index % 100 == 0:
            options['t'] += 10

if __name__ == '__main__':

    env = TiledMountainCarEnv()

    near_exit = np.zeros((3, int(NUM_EPISODES / 1000)))
    left_side = np.zeros((3, int(NUM_EPISODES / 1000)))
    x = [i for i in range(near_exit.shape[1])]

    for k, alpha in enumerate([0.1, 0.01, 0.001]):

        dt = 1.0
        options = {"t": dt}
        td_zero_semi_grad = TDSemiGred(env=env, alpha=alpha)

        # loop over the episodes and train the agent
        td_zero_semi_grad.train(**options)




