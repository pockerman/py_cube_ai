import unittest

from .test_replay_buffer import TestReplayBuffer


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestReplayBuffer)
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())