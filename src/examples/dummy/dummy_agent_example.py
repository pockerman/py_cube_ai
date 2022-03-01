import gym
import matplotlib.pyplot as plt
from src.algorithms.dummy.dummy_gym_agent import DummyGymAgent, DummyAlgoConfig
from src.worlds.gym_world_wrapper import GymWorldWrapper
from src.algorithms.rl_serial_agent_trainer import RLSerialAgentTrainer, RLSerialTrainerConfig


if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env = GymWorldWrapper(gym_env=env)

    agent_config = DummyAlgoConfig(n_itrs_per_episode=1000,
                                   render_env=True, render_env_freq=10)
    agent = DummyGymAgent(algo_config=agent_config)

    trainer_config = RLSerialTrainerConfig(n_episodes=10)
    trainer = RLSerialAgentTrainer(agent=agent, config=trainer_config)

    trainer.train(env)

    plt.plot(trainer.rewards)
    plt.show()
