from typing import TypeVar
from src.simulator.worlds.world_base import WorldBase

DefaultWorld = TypeVar("DefaultWorld")


class DefaultWorld(WorldBase):
    def __init__(self, period: float) -> None:
        super(DefaultWorld, self).__init__(period=period)

    def rebuild(self) -> DefaultWorld:
        """
        Rebuild the world
        :return:
        """
        world = DefaultWorld(period=self.period)

        world.dt = self.dt
        world.world_time = 0.0

        for robot in self.robots:
            world.add_robot(robot=robot)

        for obs in self.obstacles:
            world.add_obstacle(obstacle=obs)

        for sup in self.supervisors:
            world.supervisors.append(sup)

        return world