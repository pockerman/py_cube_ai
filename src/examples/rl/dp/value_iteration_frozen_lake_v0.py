import gym
import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.dp.value_iteration import ValueIteration, DPAlgoConfig
from src.trainers.rl_serial_algorithm_trainer import RLSerialTrainerConfig, RLSerialAgentTrainer
from src.policies.uniform_policy import UniformPolicy
from src.policies.max_action_equal_probability_stochastic_policy_adaptor import MaxActionEqualProbabilityAdaptorPolicy
from src.worlds.world_helpers import n_states, n_actions


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
    policy_init = UniformPolicy(n_actions=n_actions(env),
                                n_states=n_states(env),
                                init_val=None)
    policy_adaptor = MaxActionEqualProbabilityAdaptorPolicy()

    agent_config = DPAlgoConfig(gamma=GAMMA, tolerance=1.0e-10, policy=policy_init)

    agent = ValueIteration(agent_config, policy_adaptor)

    trainer_config = RLSerialTrainerConfig(n_episodes=1000,
                                           tolerance=1.0e-10,
                                           output_msg_frequency=100)

    trainer = RLSerialAgentTrainer(config=trainer_config, agent=agent)
    ctrl_res = trainer.train(env)

    print(f"Converged {ctrl_res.converged}")
    print(f"Number of iterations {ctrl_res.n_itrs}")
    print(f"Residual {ctrl_res.residual}")

    print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
    print(agent.policy.policy, "\n")

    plot_values(agent.v)


