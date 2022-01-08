from typing import TypeVar, List

Robot = TypeVar('Robot')
ProximitySensor = TypeVar('ProximitySensor')

def init_robot_proximity_sensors(robot: Robot, sampling_period: int):

    sensors = []
    for sensor_idx in range(8):
        # the two front distance sensors
        ps = robot.getDevice(name="ps" + str(sensor_idx))
        ps.enable(samplingPeriod=sampling_period)
        sensors.append(ps)

    return sensors

def read_proximity_sensors(sensors: List[ProximitySensor], threshold: int) -> list:

    sensor_vals = []
    crushed = False
    for i, sensor in enumerate(sensors):

        sensor_val = sensor.getValue()
        sensor_vals.append((i, sensor_val))

        if sensor_val >= threshold:
            crushed = True

    sensor_vals.append(crushed)
    return sensor_vals



