from typing import Any
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib
#import gui.frame
#import gui.viewer

from src.simulator.simulator_base import SimulatorBase
from src.simulator.default_simulator import DefaultSimulator
from src.simulator.gui.viewer import Viewer
from src.simulator.viewers.world_view import WorldView
from src.simulator.maps.map_manager import MapManager
from src.simulator.worlds.default_world import DefaultWorld
from src.simulator.robots.diff_drive_robot_base import DiffDriveRobot
from src.utils import INFO

# hertz
REFRESH_RATE = 20.0


if __name__ == '__main__':

    # the robot to simulate
    robot = DiffDriveRobot()

    viewer = Viewer(simulator=None)
    world = DefaultWorld(period=1.0/REFRESH_RATE)

    world.add_robot(robot=robot)

    world_view = WorldView(world=world, viewer=viewer)

    map_manager = MapManager()

    simulator = DefaultSimulator(refresh_rate=REFRESH_RATE, viewer=viewer,
                                 world=world, world_view=world_view, map_manager=map_manager)

    simulator.simulate()

