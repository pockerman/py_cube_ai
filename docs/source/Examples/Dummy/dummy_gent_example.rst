Example 0: ``DummyAgent`` class 
===============================

This example uses the ```DummyAlgorithm`` to train a ``DummyAgent`` agent. As its name
suggest, the ``DummyAgent`` is not really smart. However, this example illustrates some core
concepts in ``PyCubeAI``. Namely, we have three core ideas

- A trainer class (see  `Trainer specification <../../Specs/trainer_specification.html>`_)
- An algorithm to train (see `Algorithm specification <../../Specs/trainer_specification.html>`_)
- An agent that uses the output of the trained algorithm to step in the environemnt (see `Agent specification <../../Specs/trainer_specification.html>`_)

The specifications for these three concep

Let's start with the necessary imports

.. code-block:: 

	import gym
	import matplotlib.pyplot as plt
	from src.algorithms.dummy.dummy_algorithm import DummyAlgorithm, DummyAlgoConfig
	from src.agents.dummy_agent import DummyAgent
	from src.worlds.gym_world_wrapper import GymWorldWrapper
	from src.trainers.rl_serial_agent_trainer import RLSerialAgentTrainer, RLSerialTrainerConfig
	from src.utils.iteration_controller import IterationController


.. code-block::


	if __name__ == '__main__':

	    env = gym.make("MountainCar-v0")
	    env = GymWorldWrapper(gym_env=env)

	    algo_config = DummyAlgoConfig(n_itrs_per_episode=1000,
		                           render_env=True, render_env_freq=10)

	    algo = DummyAlgorithm(algo_config=algo_config)

	    trainer_config = RLSerialTrainerConfig(n_episodes=10)
	    trainer = RLSerialAgentTrainer(agent=algo, config=trainer_config)

	    trainer.train(env)

	    plt.plot(trainer.rewards)
	    plt.show()

	    agent = DummyAgent(policy=algo.policy)
	    env.reset()

	    criteria = IterationController(tol=1.0e-8, n_max_itrs=len(algo.policy))
	    agent.play(env, criteria)

