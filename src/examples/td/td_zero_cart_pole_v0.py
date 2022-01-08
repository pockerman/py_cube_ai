"""
TD(0) algorithm on CartPole-v0 algorithm. The example
is taken from Reinforcement in motion by Manning publications
"""

from src.algorithms.td.td_zero import TDZero
from src.algorithms.td.td_algorithm_base import TDAlgoInput
from src.worlds.titled_cart_pole import TiledCartPole


def policy(state) -> int:

    action = 0 if state < 5 else 1
    return action


if __name__ == '__main__':
    GAMMA = 1.0
    ALPHA = 0.1

    # create a tiled CartPole environment. The state vector
    # is discretized into 10 discrete bins
    env = TiledCartPole(version="v0", n_states=10, state_var_idx=2)

    td_algo_input = TDAlgoInput()
    td_algo_input.gamma = GAMMA
    td_algo_input.alpha = ALPHA
    td_algo_input.train_env = env
    td_algo_input.policy = policy
    td_algo_input.n_episodes = 1000
    td_algo_input.n_itrs_per_episode = 5000
    td_algo_input.output_freq = 100

    td_zero = TDZero(algo_in=td_algo_input)
    td_zero.train()

    v_function = td_zero.v_function

    for i in range(v_function.shape[0]):
        print(i, '%.3f' % v_function[i])
