from typing import Any
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib
#import gui.frame
#import gui.viewer

from src.simulator.simulator_base import SimulatorBase
from src.simulator.gui.viewer import Viewer
from src.utils import INFO


class WorldView(object):

    def __init__(self, viewer):
        self.viewer = viewer


class Simulator(SimulatorBase):

    def __init__(self, refresh_rate, world: Any, world_view: Any):
        super(Simulator, self).__init__(refresh_rate=refresh_rate, world=world, world_view=world_view)
        self.viewer = Viewer(simulator=self)
        # start Gtk
        Gtk.main()


    def simulate(self) -> None:
        """
        Simulate the world given
        :return: None
        """
        print("{} Starting simulation".format(INFO))
        self.viewer.draw_frame()
        print("{} End simulation".format(INFO))





if __name__ == '__main__':

    simulator = Simulator(refresh_rate=0.1, world=None, world_view=None)
    viewer = Viewer(simulator=None)
    simulator.simulate()

