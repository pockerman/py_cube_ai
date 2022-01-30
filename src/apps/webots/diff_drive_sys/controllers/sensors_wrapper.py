"""
Various utilities for sensor manipulation
"""
from typing import TypeVar, List

Robot = TypeVar('Robot')
ProximitySensor = TypeVar('ProximitySensor')


def init_robot_proximity_sensors(robot: Robot, sampling_period: int) -> list:
    """
    Enable the proximity sensors on the robot
    :param robot:
    :param sampling_period:
    :return:
    """

    sensors = []
    for sensor_idx in range(8):
        # the two front distance sensors
        ps = robot.getDevice(name="ps" + str(sensor_idx))
        ps.enable(samplingPeriod=sampling_period)
        sensors.append(ps)

    return sensors


def init_robot_wheel_encoders(robot: Robot, sampling_period: int) -> tuple:
    """
    initialize the robot wheel encoders
    :param robot:
    :param sampling_period:
    :return:
    """
    left_wheel_encoder = robot.getDevice("left wheel sensor")
    left_wheel_encoder.enable(samplingPeriod=sampling_period)

    right_wheel_encoder = robot.getDevice("right wheel sensor")
    right_wheel_encoder.enable(samplingPeriod=sampling_period)

    return left_wheel_encoder, right_wheel_encoder


def read_proximity_sensors(sensors: List[ProximitySensor], threshold: int) -> list:
    """
    Check the values of the proximity sensors. If a value is less
    than  threshold then we assume that the robot crushed on the obstacle
    :param sensors: The list of sensors to check
    :param threshold: The threshold size to use
    :return: A list of tuples indicating the sensor id and the read sensor value
    """

    sensor_vals = []
    crushed = False
    for i, sensor in enumerate(sensors):

        sensor_val = sensor.getValue()
        sensor_vals.append((i, sensor_val))

        if sensor_val >= threshold:
            crushed = True

    sensor_vals.append(crushed)
    return sensor_vals



