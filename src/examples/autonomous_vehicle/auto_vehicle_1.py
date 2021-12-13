import gym
import highway_env
import numpy as np
import os
import collections


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
#import matplotlib.image as mpimg
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.animation as animation
#import matplotlib

from src.algorithms.dp.value_iteration import ValueIteration
from src.policies.uniform_policy import UniformPolicy
from src.policies.stochastic_policy_adaptor import StochasticAdaptorPolicy


GAMMA = 0.99
OUT_DIR = "auto_vehicle_1_output"
CORRECT_POLICY = [3., 1., 1., 1., 1., 1., 1., 1., 1., 1., 3.,
                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                  1., 1., 1., 1., 1., 1., 1., 1.]


def plot_3d_fig(data, env, img_name, x_deg=-20, y_deg=-40, show_flag=False):

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = plt.axes(projection='3d')
    X = np.arange(0, 120, 1)
    X = np.arange(0, env.observation_space.transition.shape[0], 1)
    Y = np.arange(0, env.observation_space.transition.shape[1], 1)
    Y, X = np.meshgrid(Y, X)
    Z = data
    ax.plot_surface(X, Y, Z, cmap='magma', rstride=1, cstride=1, linewidth=0, alpha=0.7)
    ax.view_init(x_deg, y_deg)
    plt.xlabel("States")
    plt.ylabel("Actions")
    plt.savefig(OUT_DIR + '/'+img_name)

    # To switch off the display output of plot.
    if show_flag == False:
        plt.close(fig)


class ObsSpace(object):
    """
    Wrapper to the MDP to conform to the API
    expected by ValueIteration
    """

    def __init__(self, env):
        self.mdp_h = env.unwrapped.to_finite_mdp()

        print("Lane change task MDP Transition "
              "Matrix shape: " + str(self.mdp_h.transition.shape))
        print("Lane change task Reward "
              "Matrix shape: " + str(self.mdp_h.reward.shape))

    @property
    def n(self):
        return self.mdp_h.transition.shape[0]

    @property
    def transition(self):
        return self.mdp_h.transition

    @property
    def reward(self):
        return self.mdp_h.reward


class EnvWrapper(object):
    """
    Environment wrapper to conform to the environment
    API expected by ValueIteration
    """

    def __init__(self, name: str):
        self.env = gym.make(name)
        self.obs_space = ObsSpace(env=self.env)
        self.p = np.ones(shape=(self.obs_space.n, self.env.action_space.n))

        self.P = {}

        for s in range(self.obs_space.transition.shape[0]):
            self.P[s] = {}
            for a in range(self.env.action_space.n):
                self.P[s][a] = [[1.0, self.obs_space.transition[s][a], self.obs_space.reward[s][a], False]]

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.obs_space

    def reset(self):
        self.env.reset()


def determine_policy(env, v, gamma=1.0):

    policy = np.zeros(env.observation_space.n)

    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            s_ = env.observation_space.transition[s][a]
            r = env.observation_space.reward[s][a]
            q_sa[a] += (1 * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


if __name__ == '__main__':

    env = EnvWrapper(name="roundabout-v0")

    print(env.P[0])

    policy_init = UniformPolicy(env=env, init_val=None)
    policy_adaptor = StochasticAdaptorPolicy()
    value_itr = ValueIteration(n_episodes=10000, tolerance=1e-10,
                               env=env, gamma=GAMMA, policy_init=policy_init,
                               policy_adaptor=policy_adaptor)

    value_itr.train()

    optimal_value_func = value_itr.v

    policy = determine_policy(env=env, v=optimal_value_func, gamma=GAMMA)
    print("Best Policy Values Determined for the MDP.\n")
    print(policy)
    assert collections.Counter(policy.tolist()) == collections.Counter(CORRECT_POLICY), "Incorrect policy computed"

    #policy = value_itr.policy
    #print(policy.values)

    #assert collections.Counter(policy.values.tolist()) == collections.Counter(CORRECT_POLICY), "Incorrect policy computed"

    plot_3d_fig(data=env.observation_space.transition, env=env, img_name='lane_change_task_transition_matrix.png')
    plot_3d_fig(data=env.observation_space.reward, env=env, img_name='lane_change_task_reward_matrix.png')


