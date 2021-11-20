import abc


class WorldBase(metaclass=abc.ABCMeta):

    def __init__(self, period: float) -> None:

        # The period of the world
        self.period = period

        # list of robots present in the world
        self.robots = []

        # list of obstacles in the world
        self.obstacles = []

    def add_robot(self, robot) -> None:
        """
        Add a new robot into the world
        :param robot:
        :return: None
        """

        if robot is None:
            raise ValueError("Cannot add None into the robots list")

        self.robots.append(robot)

    def add_obstacle(self, obstacle) -> None:
        """
        Add a new obstacle into the world
        :param obstacle:
        :return: None
        """

        if obstacle is None:
            raise ValueError("Cannot add None into the obstacles list")

        self.obstacles.append(obstacle)
