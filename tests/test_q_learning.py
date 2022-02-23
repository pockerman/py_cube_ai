"""Unit tests for QLearning class
"""
import unittest
import pytest

from src.algorithms.td.q_learning import QLearning, TDAlgoConfig
from src.utils.exceptions import InvalidParameterValue

class TestQLearning(unittest.TestCase):

    def test_constructor(self):
        config = TDAlgoConfig(n_itrs_per_episode=1)
        qlearn = QLearning(algo_config=config)

    def test_actions_before_training_raise_zero_episodes(self):
        config = TDAlgoConfig(n_itrs_per_episode=1, n_episodes=0)
        qlearn = QLearning(algo_config=config)

        with pytest.raises(InvalidParameterValue) as e:
            qlearn.actions_before_training_begins(env=None)
            self.assertEqual("n_episodes", e.value)

    def test_actions_before_training_raise_zero_itrs(self):
        config = TDAlgoConfig(n_itrs_per_episode=0, n_episodes=1)
        qlearn = QLearning(algo_config=config)

        with pytest.raises(InvalidParameterValue) as e:
            qlearn.actions_before_training_begins(env=None)
            self.assertEqual("n_itrs_per_episode", e.value)


if __name__ == '__main__':
    unittest.main()
