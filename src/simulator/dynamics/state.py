"""
The class State. Models the state of a robot
"""

from src.simulator.dynamics.pose import Pose


class State(object):
    """
    Models the state of the system
    """

    def __init__(self, pose: Pose):
        self.pose = pose

    def __str__(self) -> str:
        return "(" + str(self.pose.x) + "," \
               + str(self.pose.y) \
               + "," \
               + str(self.pose.theta) + ")"