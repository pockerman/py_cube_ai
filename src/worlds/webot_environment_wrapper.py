"""
Wrapper for the environment to use
"""

from collections import namedtuple

from controller import Robot, Supervisor, Node
from src.worlds.time_step import TimeStep
from src.apps.webots.diff_drive_sys.controllers.action_space import ActionBase
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_proximity_sensors, read_proximity_sensors
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_wheel_encoders
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors

TIME_STEP = 64

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 3520

# the state
State = namedtuple("State", ["sensors", "motors"])

class EnvConfig(object):
    def __init__(self):
        self.dt: int = TIME_STEP
        self.bump_threshold = BUMP_THESHOLD
        self.robot_name = "E-puck"


class EnvironmentWrapper(object):

    def __init__(self, robot: Robot, robot_node: Node, config: EnvConfig):

        self.robot: Robot = robot
        self.robot_node = robot_node
        self.config: EnvConfig = config
        self.left_motor = None
        self.right_motor = None
        self.proximity_sensors = None
        self.wheel_encoders = None

        # initial translation
        self.init_translation = [0., 0., 0., ]
        self.init_rotation = [0., 1.0, 0., 0., ]

        # get the transition and rotation fields
        self.translation = robot_node.getField('translation')
        self.rotation = robot_node.getField('rotation')

    @property
    def dt(self) -> int:
        return self.config.dt

    def reset(self) -> TimeStep:
        """
        Reset the environment to the initial state. This
        is done by getting a new instance of the robot.
        If the given robot is null it simply uses the current robot
        :return:
        """

        # reset the world robot position
        self.translation.setSFVec3f(self.init_translation)
        self.rotation.setSFRotation(self.init_rotation)

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
