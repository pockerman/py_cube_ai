"""
Use REINFORCE to solve the CartPole environment
"""

import gym


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(0)
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)