"""epuck_q_learn_simple_controller controller.
See the discussion here about the formula we use
to calculate the distance from the wall:

https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo

The physical data for the model can be found here

https://cyberbotics.com/doc/guide/epuck#e-puck-model

Simulation of a differential drive system. In this simulation, we train a Q-learning
agent using the distance from the distance sensor. The environment is an empty
environment surrounded by a wall. The agent is located in the middle of he world
Its goal is to move around the perimeter of the environment without bumping on the wall.
Every time the robot bumps on the wall, the episode finishes and the agent receives a reward
of -1. Every time the agent completes a round it receives  a reward of 1.
If the agent is trapped in a corner.

The description of the epuck robot can be found at: https://cyberbotics.com/doc/guide/epuck#e-puck-model
"""

import os
from multiprocessing import Process
from collections import namedtuple

from controller import Robot, Motor, Camera, DistanceSensor, Supervisor, Field
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors
from src.apps.webots.diff_drive_sys.controllers.action_space import ActionBase
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_proximity_sensors, read_proximity_sensors
from src.algorithms.td.q_learning import QLearning
from src.utils import INFO
from src.worlds.time_step import TimeStep
from src.apps.webots.diff_drive_sys.controllers.environment_wrapper import EnvironmentWrapper, EnvConfig

# Define a variable that defines the duration of each physics step.
# This macro will be used as argument to the Robot::step function,
# and it will also be used to enable the devices.
# This duration is specified in milliseconds and it must
# be a multiple of the value in the basicTimeStep field of the WorldInfo node.

TIME_STEP = 64
SPEED_RATE_FACTOR = 0.1
WALL_REAL_HEIGHT = 0.1

MAX_SPEED = 6.28
MIN_SPEED = 0.0

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 3520

State = namedtuple("State", ["sensors", "motors"])

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)


def run(environment: EnvironmentWrapper):
    """
    Runs one episode in the environment
    :param environment:
    :return:
    """

    counter = 0
    while environment.robot.step(environment.dt) != -1:

        # check the position encoders
        print("Position left ", environment.wheel_encoders[0].getValue())
        print("Position right ", environment.wheel_encoders[1].getValue())

        # The values returned by the distance
        # sensors are scaled between 0 and 4096 (piecewise linearly to the distance).
        # While 4096 means that a big amount of light is measured (an obstacle is close)
        # and 0 means that no light is measured (no obstacle).
        ps0 = environment.proximity_sensors[0]
        ps7 = environment.proximity_sensors[7]

        # detect obstacles
        right_obstacle = ps0.getValue() > 80.0
        left_obstacle = ps7.getValue() > 80.0

        # initialize motor speeds at 50% of MAX_SPEED.
        left_speed = 0.5 * MAX_SPEED
        right_speed = 0.5 * MAX_SPEED
        # modify speeds according to obstacles

        if right_obstacle and left_obstacle:
            left_speed = 0.0
            right_speed = 0.0
        elif left_obstacle:
            # turn right
            left_speed = 0.5 * MAX_SPEED
            right_speed = -0.5 * MAX_SPEED
        elif right_obstacle:
            # turn left
            left_speed = -0.5 * MAX_SPEED
            right_speed = 0.5 * MAX_SPEED
        # write actuators inputs

        environment.left_motor.setVelocity(left_speed)
        environment.right_motor.setVelocity(right_speed)

        if left_speed == 0.0 and right_speed == 0.0:
            break

        if counter >= 1000:
            break


def controller_main():
    """
    Create the robot, establish the Environment. Train the QLearner
    :return:
    """

    # number of steps to play
    supervisor = Supervisor()
    robot_node = supervisor.getFromDef(name='qlearn_e_puck')

    if robot_node is None:
        raise ValueError("Robot node is None")

    # initial translation
    init_translation = [0., 0., 0., ]
    init_rotation = [0., 1.0,  0., 0., ]

    # get the transition and rtation fields
    translation = robot_node.getField('translation')
    rotation = robot_node.getField('rotation')

    robot = supervisor
    env_config = EnvConfig()
    env_config.robot_name = "qlearn_e_puck"
    environment = EnvironmentWrapper(robot=robot, config=env_config)

    for i in range(10000):

        print("{0} At episode={1}".format(INFO, i))
        environment.reset(robot=None)

        run(environment=environment)

        translation.setSFVec3f(init_translation)
        rotation.setSFRotation(init_rotation)


# Enter here exit cleanup code.
if __name__ == '__main__':

    controller_main()

