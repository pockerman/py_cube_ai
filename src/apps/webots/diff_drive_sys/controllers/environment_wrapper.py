"""
Wrapper for the environment to use
"""

from collections import namedtuple

from controller import Robot, Supervisor
from src.worlds.time_step import TimeStep
from src.apps.webots.diff_drive_sys.controllers.action_space import ActionBase
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_proximity_sensors, read_proximity_sensors
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_wheel_encoders
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors

TIME_STEP = 64

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 3520


class EnvConfig(object):
    def __init__(self):
        self.dt: int = TIME_STEP
        self.bump_threshold = BUMP_THESHOLD
        self.robot_name = "E-puck"


State = namedtuple("State", ["sensors", "motors"])


class EnvironmentWrapper(object):

    def __init__(self, robot: Robot, config: EnvConfig):

        self.robot: Robot = robot
        self.config: EnvConfig = config
        self.left_motor = None
        self.right_motor = None
        self.proximity_sensors = None
        self.wheel_encoders = None
        #self._init_env()

    @property
    def dt(self) -> int:
        return self.config.dt

    def reset(self, robot: Robot) -> TimeStep:
        """
        Reset the environment to the initial state. This
        is done by getting a new instance of the robot.
        If the given robot is null it simply uses the current robot
        :return:
        """

        if robot is not None:
            self.robot = robot

        # get the newly reset motors
        self.left_motor, self.right_motor = init_robot_motors(robot=self.robot, left_motor_vel=0.0, right_motor_vel=0.0)
        self.proximity_sensors = init_robot_proximity_sensors(robot=self.robot, sampling_period=self.config.dt)
        self.wheel_encoders = init_robot_wheel_encoders(robot=self.robot, sampling_period=self.config.dt)

        return TimeStep(state=None, reward=0.0, done=False, info={})

    def step(self, action: ActionBase) -> TimeStep:
        # execute the action
        action.act(self.robot, self.config.dt)

        # check if the robot crushed in the environment
        # detect obstacles
        proximity_sensor_vals = read_proximity_sensors(sensors=self.proximity_sensors, threshold=self.config.bump_threshold)

        # we may have finished because the goal was reached
        done = proximity_sensor_vals[-1]

        reward = 1.0

        if proximity_sensor_vals[-1]:
            reward = -1.0

        left_encoder_pos = self.wheel_encoders[0].getValue()
        right_encoder_pos = self.wheel_encoders[1].getValue()

        state = State(sensors=proximity_sensor_vals, motors=(left_encoder_pos, right_encoder_pos))
        time_step = TimeStep(state=state, reward=reward, done=done, info={})

        # return the distance measures from the wall
        return time_step
