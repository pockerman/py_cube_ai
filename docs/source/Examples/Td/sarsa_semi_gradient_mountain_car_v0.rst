Semi-gradient SARSA on ``MountainCar-v0``
=========================================


.. code-block::

	if __name__ == '__main__':

	    env = StateAggregationMountainCarEnv(version="v0", n_states=8 * 8 * 8)

	    lrs = [0.01, 0.1, 0.2]
	    episode_lengths = np.zeros((3, NUM_EPISODES, NUM_RUNS))

	    x = [i for i in range(episode_lengths.shape[1])]

	    for k, lr in enumerate(lrs):

		print("==================================")
		print("{0} Working with learning rate {1}".format(INFO, lr))
		print("==================================")
		# for each learning rate we do a certain number
		# of runs
		for j in range(NUM_RUNS):
		    print("{0}: run {1}".format(INFO, j))
		    policy = Policy(epsilon=1.0)

		    agent_config = SemiGradSARSAConfig(n_episodes=NUM_EPISODES,
		                                       n_itrs_per_episode=2000, policy=policy, alpha=lr,
		                                       gamma=GAMMA, dt_update_frequency=100, dt_update_factor=1.0)

		    agent = EpisodicSarsaSemiGrad(algo_config=agent_config)

		    trainer_config = RLSerialTrainerConfig(n_episodes=NUM_EPISODES, tolerance=1.0e-4, output_msg_frequency=100)
		    trainer = RLSerialAgentTrainer(trainer_config, agent)
		    trainer.train(env)

		    counters = agent.counters

		    for item in counters:
		        episode_lengths[k][item-1][j] = counters[item]
		print("==================================")
		print("==================================")

	    averaged1 = np.mean(episode_lengths[0], axis=1)
	    averaged2 = np.mean(episode_lengths[1], axis=1)
	    averaged3 = np.mean(episode_lengths[2], axis=1)

	    plt.plot(averaged1, 'r--')
	    plt.plot(averaged2, 'b--')
	    plt.plot(averaged3, 'g--')

	    plt.legend(('alpha = 0.01', 'alpha = 0.1', 'alpha = 0.2'))
	    plt.title("Episode semi-gradient SARSA (MountainCar-v0)")
	    plt.xlabel("Episode")
	    plt.xlabel("Number of iterations")
	    plt.show()
	    env.close()

