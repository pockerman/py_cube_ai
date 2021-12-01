"""
Base class for robot
"""
import abc
from typing import TypeVar, Generic

Geometry = TypeVar("Geometry")
State = TypeVar("State")


class RobotBase(metaclass=abc.ABCMeta):

    def __init__(self, name: str, geometry: Geometry, init_state: State) -> None:
        self.name = name
        self.geometry = geometry
        self.state = init_state

    @abc.abstractmethod
    def move(self, **options) -> None:
        """
        :param options:
        :return:
        """
