import unittest

from .test_replay_buffer import TestReplayBuffer
from .test_q_learning import TestQLearning


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestReplayBuffer)
    suite.addTest(TestQLearning)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

