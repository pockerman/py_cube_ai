import gym
import numpy as np

import matplotlib.pyplot as plt

from src.algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluator, DPAlgoConfig
from src.policies.uniform_policy import UniformPolicy
from src.algorithms.rl_serial_agent_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer


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

    policy_init = UniformPolicy(n_actions=env.action_space.n,
                                n_states=env.observation_space.n)

    agent_config = DPAlgoConfig()
    agent_config.gamma = 1.0
    agent_config.tolerance = 1.0e-8
    agent_config.policy = policy_init

    agent = IterativePolicyEvaluator(agent_config)

    config = RLSerialTrainerConfig()
    config.n_episodes = 100

    trainer = RLSerialAgentTrainer(agent=agent, config=config)

    ctrl_res = trainer.train(env)

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    plot_values(agent.v)
