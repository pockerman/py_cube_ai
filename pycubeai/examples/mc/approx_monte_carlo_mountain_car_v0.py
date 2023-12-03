"""
Approximate Monte Carlo learning on MountainCar-v0.
The initial implementation of the algorithm is from the
Reinforcement Learning In Motion series by Manning Publications
https://livevideo.manning.com/module/56_8_5/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/approximate-monte-carlo-predictions?
"""

import numpy as np
import matplotlib.pyplot as plt
from pycubeai.utils import INFO
from pycubeai.worlds.state_aggregation_mountain_car_env import StateAggregationMountainCarEnv, StateAggregationMountainCarBounds
from pycubeai.algorithms.mc.approximate_monte_carlo import ApproxMonteCarloConfig, ApproxMonteCarlo


class Policy(object):

    def __init__(self) -> None:
        pass

    def __call__(self, state: tuple) -> int:
        """
        Move to the left if the velocity is less than 0
        otherwise pove to the right
        :param args: 
        :param kwargs: 
        :return: 
        """
        if state[1] < 4:
            return 0
        return 2


class Model(ApproxMonteCarlo):

    def __init__(self, algo_config: ApproxMonteCarloConfig) -> None:
        super(Model, self).__init__(algo_config=algo_config)

    def actions_before_episode_begins(self, **options) -> None:

        super(Model, self).actions_before_episode_begins(**options)

        if self.current_episode_index % 1000 == 0:
            # Update the tracked performance every
            # 1000 steps

            self_k = options["k"]
            idx = self.current_episode_index // 1000
            state = self.train_env.get_state_from_obs((0.43, 0.054))
            options["near_exit"][self_k][idx] = self.state_value(state=state)
            state = self.train_env.get_state_from_obs((-1.1, 0.001))
            options["left_side"][self_k][idx] = self.state_value(state=state)
            options["dt"] += 0.1


if __name__ == '__main__':

    GAMMA = 1.0
    state_bounds = StateAggregationMountainCarBounds(car_position_space=(-1.2, 0.5),
                                                     car_velocity_space=(-0.07, 0.07))

    env = StateAggregationMountainCarEnv(version="v0", n_states=8,
                                         state_bounds=state_bounds)

    algo_config = ApproxMonteCarloConfig()
    algo_config.n_episodes = 20000
    algo_config.n_itrs_per_episode = 10000
    algo_config.gamma = 1.0
    algo_config.policy = Policy()
    algo_config.train_env = env
    algo_config.output_freq = 1000

    near_exit = np.zeros((3, int(algo_config.n_episodes / 1000)))
    left_side = np.zeros((3, int(algo_config.n_episodes / 1000)))

    x = [i for i in range(near_exit.shape[1])]

    lr_values = [0.1, 0.01, 0.001]
    for k, lr in enumerate(lr_values):

        print("{0} Working with learning rate {1}".format(INFO, lr))
        dt = 1.0
        algo_config.alpha = lr

        approx_mc = Model(algo_config=algo_config)
        approx_mc.train(**{"dt": dt,
                           "near_exit": near_exit,
                           "left_side": left_side,
                           "k": k})

    plt.subplot(221)
    plt.plot(x, near_exit[0], 'r--')
    plt.plot(x, near_exit[1], 'g--')
    plt.plot(x, near_exit[2], 'b--')
    plt.title("near exit, moving right")
    plt.subplot(222)

    plt.plot(x, left_side[0], 'r--')
    plt.plot(x, left_side[1], 'g--')
    plt.plot(x, left_side[2], 'b--')
    plt.title("left side, moving right")
    plt.legend(("eta = 0.1", "eta = 0.01",  "eta = 0.001"))
    plt.show()


    """
    images_path = Path("/home/alex/qi3/rl_python/pycubeai/examples/mc/mountain_car_imgs/")
    filenames = os.listdir(images_path)

    images = []
    for filename in filenames:
        splits = filename.split('_')
        if splits[-2] == '0' or splits[-2] == '1':
            images.append(str(images_path) + "/" + filename)

    make_gif_from_images(filenames=images,
                         gif_filename=Path("/home/alex/qi3/rl_python/pycubeai/examples/mc/gifs/mountain_car.gif"))
    """


