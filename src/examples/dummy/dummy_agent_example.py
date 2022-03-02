import gym
import matplotlib.pyplot as plt
from src.algorithms.dummy.dummy_algorithm import DummyAlgorithm, DummyAlgoConfig
from src.agents.dummy_agent import DummyAgent
from src.worlds.gym_world_wrapper import GymWorldWrapper
from src.trainers.rl_serial_agent_trainer import RLSerialAgentTrainer, RLSerialTrainerConfig
from src.utils.iteration_controller import IterationController


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
