"""
The Khepera III robot.
Some info: https://cyberbotics.com/doc/guide/khepera3
"""

from src.simulator.robots.diff_drive_robot_base import DiffDriveRobotBase
from src.simulator.dynamics.state import State
from src.simulator.dynamics.pose import Pose
from src.simulator.models.polygon import Polygon
from src.utils import INFO

# Khepera III Properties
K3_WHEEL_RADIUS = 0.021  # meters
K3_WHEEL_BASE_LENGTH = 0.0885  # meters
K3_WHEEL_TICKS_PER_REV = 2765
K3_MAX_WHEEL_DRIVE_RATE = 15.0  # rad/s

# Khepera III Dimensions
K3_BOTTOM_PLATE = [
    [-0.024, 0.064],
    [0.033, 0.064],
    [0.057, 0.043],
    [0.074, 0.010],
    [0.074, -0.010],
    [0.057, -0.043],
    [0.033, -0.064],
    [-0.025, -0.064],
    [-0.042, -0.043],
    [-0.048, -0.010],
    [-0.048, 0.010],
    [-0.042, 0.043],
]

K3_SENSOR_MIN_RANGE = 0.02
K3_SENSOR_MAX_RANGE = 0.2
K3_SENSOR_POSES = [
    [-0.038, 0.048, 128],  # x, y, theta_degrees
    [0.019, 0.064, 75],
    [0.050, 0.050, 42],
    [0.070, 0.017, 13],
    [0.070, -0.017, -13],
    [0.050, -0.050, -42],
    [0.019, -0.064, -75],
    [-0.038, -0.048, -128],
    [-0.048, 0.000, 180],
]


class KheperaIII(DiffDriveRobotBase):
    """
    KheperaIII class. Models the Khepera III robot
    """

    def __init__(self, init_state: State = State(pose=Pose(0.0, 0.0, 0.0))):
        super(KheperaIII, self).__init__(name="KheperaIII", geometry=Polygon(K3_BOTTOM_PLATE),
                                         options={"wheel_radius": K3_WHEEL_RADIUS,
                                                  "wheel_base_length": K3_WHEEL_BASE_LENGTH,
                                                  "WHEEL_TICKS_PER_REV": K3_WHEEL_TICKS_PER_REV,
                                                  "IR_SENSOR_POSES": K3_SENSOR_POSES,
                                                  "IR_SENSOR_MIN_RANGE": K3_SENSOR_MIN_RANGE,
                                                  "IR_SENSOR_MAX_RANGE": K3_SENSOR_MAX_RANGE,
                                                  "MAX_WHEEL_DRIVE_RATE": K3_MAX_WHEEL_DRIVE_RATE},
                                         init_state=init_state)

        self.global_geometry = Polygon(K3_BOTTOM_PLATE)  # actual geometry in world space

    def move(self, **options) -> None:
        """
        simulate the robot's motion over the given time interval
        :param options:
        :return: None
        """

        dt = options["dt"]
        v_l = self.left_wheel_drive_rate
        v_r = self.right_wheel_drive_rate

        # apply the robot dynamics to moving parts
        self.dynamics.apply_dynamics(v_l, v_r, dt, self.state.pose, self.wheel_encoders)

        if self.print_msgs:
            print("{0} {1} state: {2}".format(INFO, self.name, self.state))

        # update global geometry
        self.global_geometry = self.geometry.get_transformation_to_pose(self.state.pose)

        # update all of the sensors
        self.update_sensors(**options)

    def update_sensors(self, **options) -> None:
        """
        Update the robot sensors
        :param options:
        :return: None
        """

        for ir_sensor in self.ir_sensors:
            ir_sensor.update_position()



