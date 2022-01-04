"""EPuckGoForward controller.
See the discussion here about the formula we use
to calculate the distance from the wall:

https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo

The physical data for the model can be found here

https://cyberbotics.com/doc/guide/epuck#e-puck-model
"""

import cv2
from collections import deque
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Camera, DistanceSensor
from src.apps.webots.diff_drive_sys.controllers.motor_wrapper import init_robot_motors
from src.algorithms.td.q_learning import QLearning
from src.utils import INFO

MAX_SPEED = 6.28

# Define a variable that defines the duration of each physics step.
# This macro will be used as argument to the Robot::step function,
# and it will also be used to enable the devices.
# This duration is specified in milliseconds and it must
# be a multiple of the value in the basicTimeStep field of the WorldInfo node.

TIME_STEP = 64
SPEED_RATE_FACTOR = 0.1
WALL_REAL_HEIGHT = 0.1

# create the Robot instance.
robot = Robot()

camera = robot.getDevice("camera")

camera.enable(samplingPeriod=5)

camera_width = camera.getWidth()
camera_height = camera.getHeight()

print("{0} Camera width/height {1}/{2}".format(INFO, camera_width, camera_height))
byte_size = camera_width * camera_height * 4

# camera focal length
f = camera.getFocalLength()

leftMotor, rightMotor = init_robot_motors(robot=robot, left_motor_vel=0.0, right_motor_vel=0.0)

"""
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')

# set the target position of the motors
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

# set up the motor speeds at 10% of the MAX_SPEED.
leftMotor.setVelocity(0.1 * MAX_SPEED)
rightMotor.setVelocity(0.1 * MAX_SPEED)
"""

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print("{0} Time step size {1}".format(INFO, timestep))

# the two front distance sensors
ps0 = robot.getDevice(name="ps0")
ps0.enable(samplingPeriod=TIME_STEP)

ps7 = robot.getDevice(name="ps7")
ps7.enable(samplingPeriod=TIME_STEP)

print("{0} Distance sensor ps0 type {1}".format(INFO, ps0.getType()))
print("{0} Distance sensor ps1 type {1}".format(INFO, ps7.getType()))



# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    image = camera.getImage()

    # add an image to to the image buffer
    #image_buffer.append(image)

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

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    #pass

# Enter here exit cleanup code.
