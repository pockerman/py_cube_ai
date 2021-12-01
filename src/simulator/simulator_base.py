"""
Base class to derive various simulators
"""

import abc
from typing import TypeVar, Generic

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib

World = TypeVar("World")
WorldView = TypeVar("WorldView")
MapManager = TypeVar("MapManager")
Viewer = TypeVar("Viewer")


class SimulatorBase(Generic[Viewer, World, WorldView, MapManager], metaclass=abc.ABCMeta):

    def __init__(self, refresh_rate: int, viewer: Viewer, world: World,
                 world_view: WorldView, map_manager: MapManager) -> None:

        if refresh_rate < 1:
            raise ValueError("refresh_rate should be > 1")

        self.refresh_rate = refresh_rate
        self.viewer = viewer
        self.world = world
        self.world_view = world_view
        self.map_manager = map_manager
        self.period = 1.0 / refresh_rate


        if self.viewer.simulator is None:
            self.viewer.simulator = self

        # initialize views for world objects
        #self.robot_views = []

        #for robot in self.world.robots:
        #    self.add_robot_view(robot)

        #self.obstacle_views = []
        #for obstacle in self.world.obstacles:
        #    self.add_obstacle_view(obstacle)

        # Gtk simulation event source - for simulation control
        # we use this opportunity to initialize the sim
        self.sim_event_source = GLib.idle_add(self.initialize_sim)

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