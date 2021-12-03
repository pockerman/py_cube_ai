from math import radians
from src.simulator.robots.robot_base import RobotBase, Geometry
from src.simulator.dynamics.state import State
from src.simulator.dynamics.pose import Pose
from src.simulator.dynamics.differential_drive_dynamics import DifferentialDriveDynamics
from src.simulator.models.proximity_sensor import ProximitySensor
from src.simulator.models.wheel_encoder import WheelEncoder
from src.simulator.control.supervisor import Supervisor
from src.simulator.control.robot_supervisor_interface import RobotSupervisorInterface


class DiffDriveRobotBase(RobotBase):

    def __init__(self, name: str, geometry: Geometry, options: dict, init_state: State) -> None:
        super(DiffDriveRobotBase, self).__init__(name=name, geometry=geometry, init_state=init_state)
        self.wheel_radius = options["wheel_radius"] #K3_WHEEL_RADIUS  # meters
        self.wheel_base_length = options["wheel_base_length"]
        self.max_wheel_drive_rate = options["MAX_WHEEL_DRIVE_RATE"]
        self.state = init_state

        # wheel encoders
        self.left_wheel_encoder = WheelEncoder(options["WHEEL_TICKS_PER_REV"])
        self.right_wheel_encoder = WheelEncoder(options["WHEEL_TICKS_PER_REV"])
        self.wheel_encoders = [self.left_wheel_encoder, self.right_wheel_encoder]

        # initialize state
        # set wheel drive rates (rad/s)
        self.left_wheel_drive_rate = 0.0
        self.right_wheel_drive_rate = 0.0

        # IR sensors
        self.ir_sensors = []
        for _pose in options["IR_SENSOR_POSES"]:
            ir_pose = Pose(_pose[0], _pose[1], radians(_pose[2]))
            self.ir_sensors.append(ProximitySensor(self, ir_pose, options["IR_SENSOR_MIN_RANGE"],
                                                   options["IR_SENSOR_MAX_RANGE"], radians(20)))

        # dynamics
        self.dynamics = DifferentialDriveDynamics(self.wheel_radius,
                                                  self.wheel_base_length)

        # supervisor
        self.supervisor = Supervisor(RobotSupervisorInterface(self), options["wheel_radius"],
                                     options["wheel_base_length"], options["WHEEL_TICKS_PER_REV"],
                                     options["IR_SENSOR_POSES"], options["IR_SENSOR_MAX_RANGE"])

    def set_wheel_drive_rates(self, v_l, v_r):
        """
        set the drive rates (angular velocities) for this robot's wheels in rad/s
        :param v_l:
        :param v_r:
        :return:
        """
        # simulate physical limit on drive motors
        v_l = min(self.max_wheel_drive_rate, v_l)
        v_r = min(self.max_wheel_drive_rate, v_r)

        # set drive rates
        self.left_wheel_drive_rate = v_l
        self.right_wheel_drive_rate = v_r


