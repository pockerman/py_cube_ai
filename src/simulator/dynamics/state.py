"""

"""

from src.simulator.dynamics.pose import Pose


class State(object):
    """
    Models the state of the system
    """

    def __init__(self, pose: Pose):
        self.pose = pose