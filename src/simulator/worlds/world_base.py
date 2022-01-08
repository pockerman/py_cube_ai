import abc
from typing import TypeVar
from src.simulator.physics.world_physics import Physics

WorldBase = TypeVar('WorldBase')


class WorldBase(metaclass=abc.ABCMeta):

    def __init__(self, period: float, dt=0.05) -> None:

        self.physics = Physics(self)

        self.dt = dt

        # The period of the world
        self.period = period

        self.supervisors = []

        # list of robots present in the world
        self.robots = []

        # list of obstacles in the world
        self.obstacles = []

        # seconds
        self.world_time = 0.0

    @abc.abstractmethod
    def rebuild(self) -> WorldBase:
        """
        Rebuild the world
        :return:
        """

    def add_robot(self, robot) -> None:
        """
        Add a new robot into the world
        :param robot:
        :return: None
        """

        if robot is None:
            raise ValueError("Cannot add None into the robots list")

        self.robots.append(robot)
        self.supervisors.append(robot.supervisor)

    def add_obstacle(self, obstacle) -> None:
        """
        Add a new obstacle into the world
        :param obstacle:
        :return: None
        """

        if obstacle is None:
            raise ValueError("Cannot add None into the obstacles list")

        self.obstacles.append(obstacle)

        # return all objects in the world that might collide with other objects in the
        # world during simulation
    def colliders(self):
            # moving objects only
        return self.robots
             # as obstacles are static we should not test them against each other

        # return all solids in the world
    def solids(self):
        return self.robots + self.obstacles

    def step(self):

        dt = self.dt

        # step all the robots
        for robot in self.robots:
            # step robot motion
            robot.move(dt=self.dt)

        # apply physics interactions
        self.physics.apply_physics()

        # NOTE: the supervisors must run last to ensure they are observing the "current"
        # world step all of the supervisors
        for supervisor in self.supervisors:
            supervisor.on_episode(dt)

        # increment world time
        self.world_time += dt
