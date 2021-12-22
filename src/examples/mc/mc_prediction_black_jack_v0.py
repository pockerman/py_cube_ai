from typing import Any
import gym
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from algorithms.mc.mc_prediction import MCPrediction

# helpers used for plotting

def plot_blackjack_values(V):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in V:
            return V[x, y, usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


class EpisodeGenerator(object):

    def __init__(self, env:Any):
        self._env = env

    def __call__(self, *args, **kwargs):
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = self._env.reset()
        while True:
            probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
            action = np.random.choice(np.arange(2), p=probs)
            next_state, reward, done, info = self._env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        print("INFO: Generated episode with length {0}".format(len(episode)))
        return episode


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    episode_gen = EpisodeGenerator(env=env)

    mc_predictor = MCPrediction(env=env, n_max_iterations=500000, gamma=1.0, episode_generator=episode_gen)
    train_result = mc_predictor.train()
    print(train_result)

    q = mc_predictor.q

    # obtain the corresponding state-value function
    V_to_plot = dict((k, (k[0] > 18) * (np.dot([0.8, 0.2], v)) + (k[0] <= 18) * (np.dot([0.2, 0.8], v))) \
                     for k, v in q.items())

    # plot the state-value function
    plot_blackjack_values(V_to_plot)
