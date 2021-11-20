"""
Base class to derive various simulators
"""

import abc
from typing import Any


class SimulatorBase(metaclass=abc.ABCMeta):

    def __init__(self, refresh_rate, world: Any, world_view: Any):

        if refresh_rate < 1:
            raise ValueError("refresh_rate should be > 1")

        self.refresh_rate = refresh_rate
        self.world = world
        self.world_view = world_view
        self.period = 1.0 / refresh_rate

    @abc.abstractmethod
    def simulate(self) -> None:
        """
        Simulate the world given
        :return: None
        """

    @abc.abstractmethod
    def initialize_sim(self):
        """
        Initialize the data for the simulation
        :param random:
        :return:
        """



    @abc.abstractmethod
    def draw_world(self):
        """
        Draw the world
        :return:
        """