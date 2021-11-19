import unittest

from src.simulator.gui.viewer import Viewer


class TestGUIViewer(unittest.TestCase):

    def test_creation(self):
        """
        Test that we get no exception when calling the
        the constructor
        """
        try:
            viewer = Viewer(simulator=None)
            viewer.draw_frame()
        except Exception as e:
            self.fail(str(e))


if __name__ == '__main__':
    unittest.main()