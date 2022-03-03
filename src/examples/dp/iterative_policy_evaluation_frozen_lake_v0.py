"""Iterative policy evaluation on
FrozenLake-v0

"""

# Imports needed to run the example
import gym
import numpy as np
import matplotlib.pyplot as plt

# RoboRL related imports
from src.algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluator, DPAlgoConfig
from src.policies.uniform_policy import UniformPolicy
from src.trainers.rl_serial_agent_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer
from src.worlds.world_helpers import n_states, n_actions

# This is a helper function to plot the given
# value function


def plot_values(v: np.array) -> None:
    """
    Helper function to plot the given value function

    Parameters
    ----------
    v The value function to plot

    Returns
    -------

    None

    """

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

    policy_init = UniformPolicy(n_actions=n_actions(env),  n_states=n_states(env))

    agent_config = DPAlgoConfig(gamma=1.0, tolerance=1.0e-8, policy=policy_init)
    agent = IterativePolicyEvaluator(agent_config)

    config = RLSerialTrainerConfig(n_episodes=100, output_msg_frequency=10,
                                   tolerance=1.0e-8)

    trainer = RLSerialAgentTrainer(agent=agent, config=config)

    ctrl_res = trainer.train(env)

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    plot_values(agent.v)
