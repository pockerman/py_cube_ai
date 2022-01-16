"""EPuckGoForward controller.
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

from collections import namedtuple

from controller import Robot, Motor, Camera, DistanceSensor
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors
from src.apps.webots.diff_drive_sys.controllers.action_space import ActionBase
from src.apps.webots.diff_drive_sys.controllers.sensors_wrapper import init_robot_proximity_sensors, read_proximity_sensors
from src.algorithms.td.q_learning import QLearning
from src.utils import INFO
from src.worlds.time_step import TimStep

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

# create the Robot instance.
robot = Robot()

# get the motors
leftMotor, rightMotor = init_robot_motors(robot=robot, left_motor_vel=0.0, right_motor_vel=0.0)
proximity_sensors = init_robot_proximity_sensors(robot=robot, sampling_period=TIME_STEP)

# the two front distance sensors
ps0 = proximity_sensors[0]
ps7 = proximity_sensors[-1]

left_wheel_encoder = robot.getDevice("left wheel sensor")
left_wheel_encoder.enable(samplingPeriod=TIME_STEP)

right_wheel_encoder = robot.getDevice("right wheel sensor")
right_wheel_encoder.enable(samplingPeriod=TIME_STEP)

print("{0} Distance sensor ps0 type {1}".format(INFO, ps0.getType()))
print("{0} Distance sensor ps1 type {1}".format(INFO, ps7.getType()))


# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)


def step(action: ActionBase) -> TimStep:
    """
    Perform one step in the environment
    :return:
    """

    # execute the action
    action.act(robot, TIME_STEP)

    # check if the robot crushed in the environment
    # detect obstacles
    proximity_sensor_vals = read_proximity_sensors(sensors=proximity_sensors, threshold=BUMP_THESHOLD)

    # we may have finished because the goal was reached
    done = proximity_sensor_vals[-1]

    reward = 1.0

    if proximity_sensor_vals[-1]:
        reward = -1.0

    left_encoder_pos = left_wheel_encoder.getValue()
    right_encoder_pos = right_wheel_encoder.getValue()

    state = State(sensors=proximity_sensor_vals, motors=(left_encoder_pos, right_encoder_pos))
    time_step = TimStep(state=state, reward=reward, done=done, info={})

    # return the distance measures from the wall
    return time_step


def main():

    print("Running the controller")

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while robot.step(TIME_STEP) != -1:
        # Read the sensors:
        # Enter here functions to read sensor data, like:
        #  val = ds.getValue()

        print("Position left ", left_wheel_encoder.getValue())
        print("Position right ", right_wheel_encoder.getValue())

        # The values returned by the distance
        # sensors are scaled between 0 and 4096 (piecewise linearly to the distance).
        # While 4096 means that a big amount of light is measured (an obstacle is close)
        # and 0 means that no light is measured (no obstacle).

        # detect obstacles
        right_obstacle = ps0.getValue() > 80.0
        left_obstacle = ps7.getValue() > 80.0

        # initialize motor speeds at 50% of MAX_SPEED.
        leftSpeed = 0.5 * MAX_SPEED
        rightSpeed = 0.5 * MAX_SPEED
        # modify speeds according to obstacles

        if right_obstacle and left_obstacle:
            leftSpeed = 0.0
            rightSpeed = 0.0
        elif left_obstacle:
            # turn right
            leftSpeed = 0.5 * MAX_SPEED
            rightSpeed = -0.5 * MAX_SPEED
        elif right_obstacle:
            # turn left
            leftSpeed = -0.5 * MAX_SPEED
            rightSpeed = 0.5 * MAX_SPEED
        # write actuators inputs
        leftMotor.setVelocity(leftSpeed)
        rightMotor.setVelocity(rightSpeed)

        print("{0} Distance sensor ps0 type {1}".format(INFO, ps0.getValue()))
        print("{0} Distance sensor ps1 type {1}".format(INFO, ps7.getValue()))


# Enter here exit cleanup code.
if __name__ == '__main__':
    main()

