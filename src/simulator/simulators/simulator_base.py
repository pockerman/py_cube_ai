"""
Base class to derive various simulators
"""

import abc
from typing import TypeVar, Generic

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib

from src.simulator.simulator_exceptions.collision_exception import CollisionException
from src.simulator.simulator_exceptions.goal_reached_exception import GoalReachedException

World = TypeVar("World")
WorldView = TypeVar("WorldView")
MapManager = TypeVar("MapManager")
Viewer = TypeVar("Viewer")


class SimulatorBase(Generic[Viewer, World, WorldView, MapManager], metaclass=abc.ABCMeta):
    """
    Base class for deriving simulators
    """

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
    def initialize_sim(self, **options) -> None:
        """
        Initialize the data for the simulation
        :param options: Options to use in order to initialize
        the simulation
        :return: None
        """

    @abc.abstractmethod
    def draw_world(self) -> None:
        """
        Draw the world
        :return: None
        """

    def play_sim(self):

        # this ensures multiple calls to play_sim do not speed up the simulator
        GLib.source_remove(self.sim_event_source)
        self._run_sim()
        self.viewer.control_panel_state_playing()

    def pause_sim(self):
        GLib.source_remove(self.sim_event_source)
        self.viewer.control_panel_state_paused()

    def step_sim_once(self):
        self.pause_sim()
        self._step_sim()

    def end_sim(self, alert_text=""):
        GLib.source_remove(self.sim_event_source)
        self.viewer.control_panel_state_finished(alert_text)

    def reset_sim(self):
        self.pause_sim()
        self.initialize_sim()

    def save_map(self, filename):
        self.map_manager.save_map(filename)

    def load_map(self, filename):
        self.map_manager.load_map(filename)
        self.reset_sim()

    def random_map(self):
        self.pause_sim()
        self.initialize_sim(**{"random":True})

    def _run_sim(self):
        self.sim_event_source = GLib.timeout_add(int(self.period * 1000), self._run_sim)
        self._step_sim()

    def _step_sim(self):
        # increment the simulation
        try:
            self.world.step()
        except CollisionException:
            self.end_sim("Collision!")
        except GoalReachedException:
            self.end_sim("Goal Reached!")

        # draw the resulting world
        self.draw_world()