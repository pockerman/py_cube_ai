"""epuck_controller_dev_101 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
import math
import numpy as np

from controller import Robot

TIME_STEP = 32


def main_0(robot: Robot, limit: int = 4) -> None:

    counter = 0
    while robot.step(TIME_STEP) != -1:
        print("Hello  from robot {0}".format(robot.getName()))
        counter += 1
        if counter >= limit:
            break


def main_1(robot: Robot, limit: int = 4) -> None:

    sensor_ps0 = robot.getDevice("ps0")
    sensor_ps0.enable(TIME_STEP)

    counter = 0
    while robot.step(TIME_STEP) != -1:
        print("Hello  from robot {0}".format(robot.getName()))
        print("Sensor value {0}".format(sensor_ps0.getValue()))
        counter += 1
        if counter >= limit:
            break


def main_2(robot: Robot, limit: int = 4) -> None:

    left_motor = robot.getDevice("left wheel motor")
    F = 2.0
    t = 0.0
    counter = 0
    while robot.step(TIME_STEP) != -1:
        print("Hello  from robot {0}".format(robot.getName()))

        position = math.sin(t * 2.0 * np.pi * F)
        left_motor.setPosition(position)

        t += TIME_STEP / 1000.0
        counter += 1
        if counter >= limit:
            break


if __name__ == '__main__':

    # create the Robot instance.
    robot = Robot()

    # main_0(robot=robot)
    #main_1(robot=robot)
    main_2(robot=robot)

