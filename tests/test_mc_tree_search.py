import unittest

from src.algorithms.planning.monte_carlo_tree_search import MCTreeSearch
from src.algorithms.algo_input import AlgoInput


class TestMCTreeSearch(unittest.TestCase):

    def test_ctor(self):
        """
        Test that an instance can be created
        """
        try:
            input = AlgoInput()
            obj = MCTreeSearch(input=input)
        except Exception as e:
            print("An exception was thrown whilst constructing the object.")
            print("Exception message {}".format(str(e)))
            raise e


if __name__ == '__main__':
    unittest.main()