QLearning on ``CartPole-v0`` (Python)
=====================================

In this example we use Q-learning to train an agent on the ``CartPole-v0`` environment.
The state for this environment exposed to the agent, consists of four floating point
values. Thus, for this state we cannot use a tabular based method. Therefore, we employ
the wrapper class ``StateAggregationCartPoleEnv`` that uses bins to cast each variable
in the state vector as an integer.

.. code-block::

	def plot_running_avg(avg_rewards):

	    running_avg = np.empty(avg_rewards.shape[0])
	    for t in range(avg_rewards.shape[0]):
		running_avg[t] = np.mean(avg_rewards[max(0, t-100) : (t+1)])
	    plt.plot(running_avg)
	    plt.xlabel("Number of episodes")
	    plt.ylabel("Reward")
	    plt.title("Running average")
	    plt.show()
	    
.. code-block::

	if __name__ == '__main__':
	    GAMMA = 1.0
	    ALPHA = 0.1
	    EPS = 1.0

	    env = StateAggregationCartPoleEnv(n_states=10)

	    agent_config = TDAlgoConfig(gamma=GAMMA, alpha=ALPHA,
		                        n_itrs_per_episode=50000,
		                        n_episodes=10000,
		                        policy=EpsilonGreedyPolicy(n_actions=n_actions(env),
		                                                   eps=EPS, decay_op=EpsilonDecayOption.INVERSE_STEP))

	    agent = QLearning(agent_config)

	    trainer_config = RLSerialTrainerConfig(n_episodes=50000, output_msg_frequency=5000)
	    trainer = RLSerialAgentTrainer(trainer_config, agent=agent)
	    trainer.train(env)

	    plot_running_avg(agent.total_rewards)

