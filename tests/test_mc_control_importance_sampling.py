import unittest
from algorithms.mc.mc_control_importance_sampling import MCControlImportanceSampling
from algorithms.policy_sampler import DummyPolicySampler


class TestMCControlImportanceSampling(unittest.TestCase):

    def test_creation(self):
        """
        Test that we get no exception when calling the
        the constructor
        """
        try:
            agent = MCControlImportanceSampling(n_episodes=100, itrs_per_episode=150,
                                                discount_factor=0.9,
                                                behavior_policy=DummyPolicySampler(),
                                                tolerance=0.01)
        except Exception as e:
            self.fail(str(e))


if __name__ == '__main__':
    unittest.main()