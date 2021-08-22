import gym
import numpy as np
from typing import Any
import matplotlib.pyplot as plt

from algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluator
from policies.uniform_policy import UniformPolicy


class Agent(IterativePolicyEvaluator):

    def __init__(self, env: Any, n_max_itrs: int,
                 tolerance: float, gamma: float, polic_init: Any) -> None:
        super(Agent, self).__init__(n_max_iterations=n_max_itrs, tolerance=tolerance,
                                    env=env, policy_init=polic_init, gamma=gamma)


def plot_values(v):
    # reshape value function
    V_sq = np.reshape(v, (4, 4))

    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title('State-Value Function')
    plt.show()


if __name__ == '__main__':
    env = gym.make("FrozenLake-v0")
    policy_init = UniformPolicy(env=env)
    agent = Agent(env=env, n_max_itrs=1, gamma=1.0,
                  tolerance=1.0e-8, polic_init=policy_init)

    ctrl_res = agent.train()

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    plot_values(agent.v)
