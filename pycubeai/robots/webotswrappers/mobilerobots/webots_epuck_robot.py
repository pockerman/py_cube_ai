"""Module webots_epuck_robot. Wrapper to
Webots Epuck robot

"""

from dataclasses import dataclass
import numpy as np
from typing import TypeVar

from pycubeai.robots.webots_mobile_robot_base import WebotsMobileRobotBase

WebotRobotType = TypeVar('WebotRobotType')
WebotNodeType = TypeVar('WebotNodeType')

@dataclass(init=True, repr=True)
class WebotsEpuckRobotPose(object):

    position: np.array = np.zeros(3)
    orientation: np.array = np.array([0.0, 1.0, 0.0, 0.0])


@dataclass(init=True, repr=True)
class WebotsEpuckRobotConfiguration(object):
    sampling_period: int = 32
    init_proximity_sensors: bool = True
    robot: WebotRobotType = None
    robot_node: WebotNodeType = None


class WebotsEpuckRobot(WebotsMobileRobotBase):

    def __init__(self, config: WebotsEpuckRobotConfiguration,
                 pose: WebotsEpuckRobotPose=WebotsEpuckRobotPose()):
        super(WebotsEpuckRobot, self).__init__(name="webots-epuck", pose=pose)
        self.config: WebotsEpuckRobotConfiguration = config
        self.proximity_sensors = {}
        self._wheel_motor_idx_to_name = {0: 'left wheel motor', 1: 'right wheel motor'}


    def n_motors(self) -> int:
        return 2

    def n_wheels(self) -> int:
        return 2

    def position(self) -> np.ndarray:
        return self.pose.position

    def orientation(self) -> float:
        pass


    def read_proximity_sensor(self, name: str) -> float:
        """Read the proximity sensor with the given name

        Parameters
        ----------
        name: Name of the proximity sensor

        Returns
        -------
        The value read by the proximity sensor
        """
        return self.proximity_sensors[name].getValue()

    def set_wheel_motor_position(self, motor_idx: int, motor_pos: float) -> None:
        """Set the position of the left motor

        Parameters
        ----------
        motor_pos: The value to set the motor position
        motor_idx: The motor index to set

        Returns
        -------

        """
        motor_name = self._wheel_motor_idx_to_name[motor_idx]
        self.config.robot.getDevice(motor_name).setPosition(motor_pos)

    def set_wheel_motor_velocity(self, motor_idx: int, motor_vel: float) -> None:
        motor_name = self._wheel_motor_idx_to_name[motor_idx]
        self.config.robot.getDevice(motor_name).setVelocity(motor_vel)

    def init_proximity_sensors(self) -> None:

        """Enable the proximity sensors on the robot

        Returns
        -------
        None

        """

        for sensor_idx in range(8):
            # the two front distance sensors
            name = "ps" + str(sensor_idx)
            ps = self.config.robot.getDevice(name=name)
            ps.enable(samplingPeriod=self.config.sampling_period)
            self.proximity_sensors[name] = ps


