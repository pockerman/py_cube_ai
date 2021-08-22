import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from algorithms.dp.value_iteration import ValueIteration
from policies.uniform_policy import UniformPolicy
from policies.stochastic_policy_adaptor import StochasticAdaptorPolicy


class Agent(ValueIteration):

    def __init__(self, env: Any, gamma: float, policy_init: UniformPolicy,
                 policy_adaptor: StochasticAdaptorPolicy, n_max_iterations: int=1000, tolerance: float=1.0e-10) -> None:
        super(Agent, self).__init__(env=env, gamma=gamma, policy_init=policy_init,
                                    policy_adaptor=policy_adaptor,
                                    n_max_iterations=n_max_iterations, tolerance=tolerance)
        
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

    ENV_NAME = "FrozenLake-v0"
    GAMMA = 1.0

    env = gym.make(ENV_NAME)
    policy_init = UniformPolicy(env=env, init_val=None)
    policy_adaptor = StochasticAdaptorPolicy()
    agent = Agent(gamma=GAMMA, env=env,
                  policy_init=policy_init, policy_adaptor=policy_adaptor)

    ctrl_res = agent.train()

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(agent.policy.values, "\n")

    plot_values(agent.v)


