import gym
import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.td import QLearning
from src.policies.epsilon_greedy_policy import EpsilonDecayOption, EpsilonGreedyPolicy


def plot_values(V):
    # reshape the state-value function
    V = np.reshape(V, (4, 12))
    # plot the state-value function
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(V, cmap='cool')
    for (j, i), label in np.ndenumerate(V):
        ax.text(i, j, np.round(label, 3), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title('State-Value Function')
    plt.show()


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')

    num_episodes = 5000
    plot_every = 100
    policy = EpsilonGreedyPolicy(eps=1.0, env=env,
                                 decay_op=EpsilonDecayOption.INVERSE_STEP,
                                 min_eps=0.0001)
    q_learner = QLearning(env=env, n_max_iterations=num_episodes, gamma=1.0, alpha=0.01,
                          plot_freq=plot_every, policy=policy, max_num_iterations_per_episode=1000,
                          tolerance=1.0e-4)

    q_learner.train()

    q_func = q_learner.q_function

    # print the estimated optimal policy
    policy_q_learning = np.array(
        [np.argmax(q_func[key]) if key in q_func else -1 for key in np.arange(48)]).reshape((4, 12))

    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_q_learning)

    # plot performance
    plt.plot(np.linspace(0, num_episodes, len(q_learner.avg_scores), endpoint=False), np.asarray(q_learner.avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()

    # plot the estimated optimal state-value function
    plot_values([np.max(q_func[key]) if key in q_func else 0 for key in np.arange(48)])
