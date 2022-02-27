from typing import TypeVar

Robot = TypeVar('Robot')


def init_robot_motors(robot: Robot, left_motor_vel: float, right_motor_vel: float):

    left_motor = robot.getDevice('left wheel motor')
    right_motor = robot.getDevice('right wheel motor')

    # set the target position of the motors
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))

    # set up the motor speeds at 10% of the MAX_SPEED.
    left_motor.setVelocity(left_motor_vel)
    right_motor.setVelocity(right_motor_vel)

    return left_motor, right_motor