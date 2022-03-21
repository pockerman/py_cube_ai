
Q-learning on ``CliffWalking-v0`` (Python)
==========================================

Overview
--------

Code
----

.. code-block::

	import gym
	import numpy as np
	import matplotlib.pyplot as plt
	from src.algorithms import QLearning
	from src.algorithms import TDAlgoConfig
	from src.worlds import GymWorldWrapper
	from src.trainers import RLSerialAgentTrainer, RLSerialTrainerConfig
	from src.policies.epsilon_greedy_policy import EpsilonDecayOption, EpsilonGreedyPolicy

.. code-block::

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

.. code-block::

	if __name__ == '__main__':

	    train_env = GymWorldWrapper(gym.make('CliffWalking-v0'))

	    ALPHA = 0.01
	    GAMMA = 1.0
	    N_ITRS_EPISODES = 1000
	    N_EPISODES = 5000
	    EPS = 1.0
	    plot_every = 100

	    agent_config = TDAlgoConfig(gamma=GAMMA, alpha=ALPHA,
		                        n_itrs_per_episode=N_ITRS_EPISODES,
		                        n_episodes=N_EPISODES,
		                        policy=EpsilonGreedyPolicy(n_actions=train_env.n_actions,
		                                                   eps=EPS, decay_op=EpsilonDecayOption.INVERSE_STEP))

	    q_learner = QLearning(agent_config)

	    trainer_config = RLSerialTrainerConfig(n_episodes=N_EPISODES, output_msg_frequency=100)
	    trainer = RLSerialAgentTrainer(config=trainer_config, algorithm=q_learner)

	    trainer.train(train_env)

	    q_func = q_learner.q_function

	    # print the estimated optimal policy
	    policy_q_learning = np.array(
		[np.argmax(q_func[key]) if key in q_func else -1 for key in np.arange(48)]).reshape((4, 12))

	    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
	    print(policy_q_learning)

	    # plot performance
	    plt.plot(np.linspace(0, N_EPISODES, len(trainer.avg_rewards), endpoint=False), np.asarray(trainer.avg_rewards))
	    plt.xlabel('Episode Number')
	    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
	    plt.show()

	    # plot the estimated optimal state-value function
	    plot_values([np.max(q_func[key]) if key in q_func else 0 for key in np.arange(48)])
