from src.simulator.robots.robot_base import RobotBase, Geometry


class DiffDriveRobotBase(RobotBase):

    def __init__(self, name: str, geometry: Geometry, options: dict) -> None:
        super(DiffDriveRobotBase, self).__init__(name=name, geometry=geometry)
        self.wheel_radius = options["wheel_radius"] #K3_WHEEL_RADIUS  # meters
        self.wheel_base_length = options["wheel_base_length"]
