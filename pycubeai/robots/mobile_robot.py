"""module mobile_robot Base class for ground mobile robots.
These robots are assumed to move in the plain therefore poses
an x,y position and an orientation theta. The dynamics is responsible
to compute these variables. Examples of robots extending this class
are:

"""
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import TypeVar
from typing import Union
from robot_base import RobotBase

from pycubeai.robots import MobileRobotPose


class MobileRobotBase(RobotBase, ABC):
    def __init__(self, name: str, specification: Union[Path, str, None],
                 pose: MobileRobotPose):
        super(MobileRobotBase, self).__init__(name=name,
                                              specification=specification)
        self._pose: MobileRobotPose = pose

    @property
    def pose(self) -> MobileRobotPose:
        return self._pose

    @abstractmethod
    def n_motors(self) -> int:
        pass

    @abstractmethod
    def n_wheels(self) -> int:
        pass

    @abstractmethod
    def position(self) -> np.ndarray:
        pass

    @abstractmethod
    def orientation(self) -> float:
        pass

    @abstractmethod
    def set_wheel_motor_position(self, motor_idx: int, motor_pos: float) -> None:
        """Set the position of the i-th motor

        Parameters
        ----------
        motor_pos: The value to set the motor position
        motor_idx: The motor index to set

        Returns
        -------

        """
        pass

    @abstractmethod
    def set_wheel_motor_velocity(self, motor_idx: int, motor_vel: float) -> None:
        """Set the velocity of the i-th motor wheel

        Parameters
        ----------
        motor_vel: The value to set the motor velocity
        motor_idx: The motor index to set
        Returns
        -------

        """
        pass

