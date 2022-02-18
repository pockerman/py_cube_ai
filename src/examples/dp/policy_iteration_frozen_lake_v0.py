import gym
import numpy as np
from typing import Any
import matplotlib.pyplot as plt

from src.algorithms.dp.policy_iteration import PolicyIteration
from src.policies.uniform_policy import UniformPolicy
from src.policies.max_action_policy_adaptor import MaxActionPolicyAdaptor
from src.worlds.world_helpers import n_actions
from src.algorithms.rl_serial_agent_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer


class Agent(PolicyIteration):

    def __init__(self, env: Any, n_max_itrs: int, n_policy_eval_steps: int,
                 tolerance: float, gamma: float,
                 polic_init: UniformPolicy,
                 policy_adaptor: StochasticAdaptorPolicy) -> None:
        super(Agent, self).__init__(n_max_iterations=n_max_itrs,
                                    n_policy_eval_steps=n_policy_eval_steps,
                                    tolerance=tolerance,
                                    env=env, policy_init=polic_init, gamma=gamma,
                                    policy_adaptor=policy_adaptor)


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
    policy_init = UniformPolicy(n_actions=n_actions(env),
                                ninit_val=None)
    policy_adaptor = MaxActionPolicyAdaptor()

    agent = Agent(env=env, n_max_itrs=100, n_policy_eval_steps=100,
                  gamma=1.0,
                  tolerance=1.0e-7, polic_init=policy_init,
                  policy_adaptor=policy_adaptor)

    config = RLSerialTrainerConfig()
    config.n_episodes = 100

    trainer = RLSerialAgentTrainer(agent=agent, config=config)

    ctrl_res = trainer.train(env)

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(agent.policy.values, "\n")

    plot_values(agent.v)
