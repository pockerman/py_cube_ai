"""
Approximate Monte Carlo learning on MountainCar-v0.
The initial implementation of the algorithm is from the
Reinforcement Learning In Motion series by Manning Publications
https://livevideo.manning.com/module/56_8_5/reinforcement-learning-in-motion/climbing-the-mountain-with-approximation-methods/approximate-monte-carlo-predictions?
"""

import gym
import os
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from src.utils import INFO
from src.utils.image_utils import make_gif, make_gif_from_images


class Policy(object):

    def __init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs) -> int:
        """
        Move to the left if the velocity is less than 4
        otherwise pove to the right
        :param args: 
        :param kwargs: 
        :return: 
        """
        if args[0] < 4:
            return 0
        return 2


def aggregate_state(pos_bins, vel_bins, obs) -> tuple:

    pos = int(np.digitize(obs[0], pos_bins))
    vel = int(np.digitize(obs[1], vel_bins))
    return (pos, vel)


class Model(object):

    def __init__(self, eta: float, state_space: List[tuple]) -> None:
        self.eta = eta
        self.weights = {}
        self.state_space = state_space
        self._init_weights()

    def state_value(self, state: tuple) -> float:
        return self.weights[state]

    def update_weights(self, total_return, state, t) -> None:
        """
        Update the model weghts. It decreases the learning rate
        as new_eta = eta/t
        :param total_return:
        :param state:
        :param t:
        :return:
        """
        self.weights[state] += self.eta/t * (total_return - self.state_value(state))

    def _init_weights(self):
        for s in self.state_space:
            self.weights[s] = 0


if __name__ == '__main__':

    GAMMA = 1.0
    env = gym.make('MountainCar-v0')

    pos_bins = np.linspace(-1.2, 0.5, 8)
    vel_bins = np.linspace(-0.07, 0.07, 8)

    state_space = [(i, j) for i in range(1, 9) for j in range(1, 9)]

    num_episodes = 20000

    near_exit = np.zeros((3, int(num_episodes / 1000)))
    left_side = np.zeros((3, int(num_episodes / 1000)))

    x = [i for i in range(near_exit.shape[1])]

    lr_values = []
    #lr_values = [0.1, 0.01, 0.001]
    for k, lr in enumerate(lr_values):
        dt = 1.0
        model = Model(eta=lr, state_space=state_space)
        policy = Policy()

        for i in range(num_episodes):
            if i % 1000 == 0:
                # Update the tracked performance every
                # 1000 steps
                print("{0} working on episdoe {1}/{2}".format(INFO, i, num_episodes))

                idx = i // 1000
                state = aggregate_state(pos_bins=pos_bins, vel_bins=vel_bins, obs=(0.43, 0.054))
                near_exit[k][idx] = model.state_value(state=state)
                state = aggregate_state(pos_bins=pos_bins, vel_bins=vel_bins, obs=(-1.1, 0.001))
                left_side[k][idx] = model.state_value(state=state)
                dt += 0.1
            observation = env.reset()
            done = False
            memory = []
            counter = 0
            while not done:
                state = aggregate_state(pos_bins=pos_bins, vel_bins=vel_bins, obs=observation)
                action = policy(*[state[1]])

                next_obs, reward, done, _ = env.step(action)
                memory.append((state, action, reward))
                observation = next_obs

                if k == 0:
                    screen = env.render(mode="rgb_array")

                    output_directory = "/home/alex/qi3/rl_python/src/examples/mc/mountain_car_imgs/"
                    fname = os.path.join(output_directory, "mountain_car_" + str(i) + "_" + str(counter) + ".png")
                    print(fname)
                    plt.imsave(fname=fname, arr=screen, format='png')
                counter += 1

            state = aggregate_state(pos_bins=pos_bins, vel_bins=vel_bins, obs=observation)
            memory.append((state, action, reward))

            G = 0
            last = True
            states_returns = []

            for state, action, reward in reversed(memory):
                if last:
                    last = False
                else:
                    states_returns.append((state, G))

                G = GAMMA * G + reward
            states_returns.reverse()
            states_visited = []
            for state, G, in states_returns:
                if state not in states_visited:
                    model.update_weights(total_return=G, state=state, t=dt)
                    states_visited.append(state)


    """
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

    images_path = Path("/home/alex/qi3/rl_python/src/examples/mc/mountain_car_imgs/")
    filenames = os.listdir(images_path)

    images = []
    for filename in filenames:
        splits = filename.split('_')
        if splits[-2] == '0' or splits[-2] == '1':
            images.append(str(images_path) + "/" + filename)
        else:
            print(splits)

    make_gif_from_images(filenames=images,
                         gif_filename=Path("/home/alex/qi3/rl_python/src/examples/mc/gifs/mountain_car.gif"))

    #make_gif(images_path=Path("/home/alex/qi3/rl_python/src/examples/mc/mountain_car_imgs/"),
    #         gif_filename=Path("/home/alex/qi3/rl_python/src/examples/mc/mountain_car.gif"))
