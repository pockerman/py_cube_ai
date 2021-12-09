import numpy as np
from src.algorithms.td.td_zero_semi_gradient import TDZeroSemiGrad

def get_bins(n_bins: int=8, n_layers: int=8):

    pos_tile_width = (0.5 + 1.2) / n_bins * 0.5
    vel_tile_width = (0.07 + 0.07) / n_bins * 0.5
    pos_bins = np.zeros((n_layers, n_bins))
    vel_bins = np.zeros((n_layers, n_bins))

    for i in range(n_layers):
        pos_bins[i] = np.linspace(-1.2 + i * pos_tile_width, 0.5 + i * pos_tile_width/2, n_bins)
        vel_bins[i] = np.linspace(-1.2 + i * vel_tile_width, 0.5 + i * vel_tile_width / 2, n_bins)

    return pos_bins, vel_bins


def tile_state(pos_bins, vel_bins, obs, n_tiles, n_layers):
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


if __name__ == '__main__':
    pass





