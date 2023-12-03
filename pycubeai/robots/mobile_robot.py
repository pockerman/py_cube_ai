"""module mobile_robot Base class for ground mobile robots.
These robots are assumed to move in the plain therefore poses
an x,y position and an orientation theta. The dynamics is responsible
to compute these variables. Examples of robots extending this class
are:

"""
from robot_base import RobotBase
from pathlib import Path
from typing import Callable
from typing import Union

class MobileRobotBase(RobotBase):
    def __init__(self, name: str, specification: Union[Path, str, None], dynamics: Callable):
        super(MobileRobotBase, self).__init__(name=name,
                                              specification=specification)
        self._dynamics = dynamics