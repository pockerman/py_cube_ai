"""
Base class to derive various simulators
"""

import abc
from typing import Any


class SimulatorBase(metaclass=abc.ABCMeta):

    def __init__(self, refresh_rate, world: Any, world_view: Any):
        self.refresh_rate = refresh_rate
        self.world = world
        self.world_view = world_view

    @abc.abstractmethod
    def simulate(self) -> None:
        """
        Simulate the world given
        :return: None
        """