"""Module webots_epuck_robot. Wrapper to
Webots Epuck robot

"""

from dataclasses import dataclass
import numpy as np
from controller import Supervisor, Node, Robot


@dataclass(init=True, repr=True)
class WebotsEpuckRobotPose(object):

    position: np.array = np.zeros(3)
    orientation: np.array = np.array([0.0, 1.0, 0.0, 0.0])


@dataclass(init=True, repr=True)
class WebotsEpuckRobotConfiguration(object):
    sampling_period: int = 32
    init_proximity_sensors: bool = True
    robot: Robot = None
    robot_node: Node = None


class WebotsEpuckRobot(object):

    def __init__(self, config: WebotsEpuckRobotConfiguration):
        self.config: WebotsEpuckRobotConfiguration = config
        self.pose: WebotsEpuckRobotPose = WebotsEpuckRobotPose()
        self.proximity_sensors = {}

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

    def set_left_wheel_motor_position(self, motor_pos: float) -> None:
        """Set the position of the left motor

        Parameters
        ----------
        motor_pos: The value to set the motor position

        Returns
        -------

        """
        left_motor = self.config.robot.getDevice('left wheel motor')
        left_motor.setPosition(motor_pos)

    def set_right_wheel_motor_position(self, motor_pos: float) -> None:
        """Set the position of the left motor

        Parameters
        ----------
        motor_pos: The value to set the motor position

        Returns
        -------

        """
        right_motor = self.config.robot.getDevice('right wheel motor')
        right_motor.setPosition(motor_pos)

    def set_left_wheel_motor_velocity(self, motor_vel: float) -> None:
        """Set the position of the left motor

        Parameters
        ----------
        motor_vel: The value to set the motor velocity

        Returns
        -------

        """
        left_motor = self.config.robot.getDevice('left wheel motor')
        left_motor.setVelocity(motor_vel)

    def set_right_wheel_motor_velocity(self, motor_vel: float) -> None:
        """Set the position of the left motor

        Parameters
        ----------
        motor_vel: The value to set the motor velocity

        Returns
        -------

        """
        right_motor = self.config.robot.getDevice('right wheel motor')
        right_motor.setVelocity(motor_vel)

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




