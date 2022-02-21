"""
Wrapper for the environment to use
"""

from collections import namedtuple
from typing import TypeVar
import numpy as np

from controller import Robot, Node
from src.worlds.time_step import TimeStep
from src.worlds.webot_robot_action_space import WebotRobotActionBase, WebotRobotActionSpace
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_proximity_sensors, read_proximity_sensors
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_wheel_encoders
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors
from src.utils import INFO

TIME_STEP = 64

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 3520

# The State nametuple wraps the information
# that we collect from the Webot simulator and send
# to the agent
State = namedtuple("State", ["position", "orientation", "velocity", "sensors", "motors"])
Criterion = TypeVar('Criterion')


class EnvConfig(object):
    def __init__(self):
        self.dt: int = TIME_STEP
        self.bump_threshold = BUMP_THESHOLD
        self.robot_name = "E-puck"
        self.on_goal_criterion: Criterion = None
        self.reward_on_wall_crush = -1.0


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

        self.action_space = WebotRobotActionSpace()

    @property
    def dt(self) -> int:
        return self.config.dt

    @property
    def n_actions(self) -> int:
        return len(self.action_space)

    @property
    def actions(self) -> list:
        return list(self.action_space.actions.keys())

    def add_action(self, action: WebotRobotActionBase) -> None:
        """
        Add a new action in the environment
        :param action:
        :return:
        """
        self.action_space.add_action(action=action)

    def get_action(self, aidx) -> WebotRobotActionBase:
        return self.action_space[aidx]

    def continue_sim(self) -> bool:
        """
        Returns true if the simulation is continued.
        We need to call robot step to synchronize the
        robot sensors etc
        :return:
        """
        return self.robot.step(duration=self.dt) != -1

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

        left_encoder_pos = self.wheel_encoders[0].getValue()
        right_encoder_pos = self.wheel_encoders[1].getValue()

        state = State(position=self.robot_node.getPosition(),
                      velocity=self.robot_node.getVelocity(),
                      orientation=self.robot_node.getOrientation(),
                      sensors={"proximity_sensors": self.proximity_sensors},
                      motors=(left_encoder_pos, right_encoder_pos))

        return TimeStep(state=state, reward=0.0, done=False, info={})

    def get_reward(self, action: WebotRobotActionBase) -> tuple:

        """
        Returns the reward associated with the action
        :param action:
        :return:
        """

        # check if the robot crushed in the environment
        # detect obstacles
        proximity_sensor_vals = read_proximity_sensors(sensors=self.proximity_sensors,
                                                       threshold=self.config.bump_threshold)

        # we may have finished because the goal was reached
        # however if we have crushed this is serious
        # so first check this
        done_crush = proximity_sensor_vals[-1]

        if done_crush:
            print("{0} Robot crushed on the wall...".format(INFO))
            print("{0} Robot position  {1}".format(INFO, self.robot_node.getPosition()))
            print("{0} Proximity sensors values {1}".format(INFO, proximity_sensor_vals))
            print("{0} Bump threshold {1}".format(INFO, self.config.bump_threshold))
            return self.config.reward_on_wall_crush, True

        # we haven't crush check if we reached the goal
        # the goal may depend on the action
        done, reward, distance = self.config.on_goal_criterion.check(self.robot_node, action)

        if done:
            print("{0} Robot reached the goal with distance from it {1}".format(INFO, distance))
            print("{0} Robot position  {1}".format(INFO, self.robot_node.getPosition()))
            return reward, True

        # we haven't crushed and haven't reached goal
        return reward, False

    def step(self, action: WebotRobotActionBase) -> TimeStep:

        # execute the action
        action.act(self.robot, self.config.dt)

        reward, done = self.get_reward(action=action)

        left_encoder_pos = self.wheel_encoders[0].getValue()
        right_encoder_pos = self.wheel_encoders[1].getValue()

        velocity = self.robot_node.getVelocity()
        #print("{0} Robot velocity {1}".format(INFO, np.linalg.norm(np.array([0., 0., 0.]) - np.array(velocity[0:3]))))
        state = State(position=self.robot_node.getPosition(),
                      velocity=self.robot_node.getVelocity(),
                      orientation=self.robot_node.getOrientation(),
                      sensors={"proximity_sensors": self.proximity_sensors},
                      motors=(left_encoder_pos, right_encoder_pos))

        time_step = TimeStep(state=state, reward=reward, done=done, info={})

        # return the distance measures from the wall
        return time_step
