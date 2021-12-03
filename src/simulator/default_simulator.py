import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib

from src.utils import INFO
from src.simulator.simulator_base import SimulatorBase, World, WorldView, Viewer, MapManager
from src.simulator.simulator_exceptions.collision_exception import CollisionException
from src.simulator.simulator_exceptions.goal_reached_exception import GoalReachedException


class DefaultSimulator(SimulatorBase):

    def __init__(self,  refresh_rate: int, viewer: Viewer, world: World,
                 world_view: WorldView, map_manager: MapManager) -> None:
        super(DefaultSimulator, self).__init__(refresh_rate=refresh_rate, viewer=viewer,
                                               world=world, world_view=world_view, map_manager=map_manager)


    def simulate(self) -> None:
        """
        Simulate the world given
        :return: None
        """
        print("{} Starting simulation".format(INFO))
        Gtk.main()
        self.viewer.draw_frame()
        print("{} End simulation".format(INFO))

    def initialize_sim(self):
        """
        Initialize the data for the simulation
        :param random:
        :return:
        """
        self.viewer.control_panel_state_init()
        self.draw_world()
        self.map_manager.random_map(self.world)

    def draw_world(self):

        # start a fresh frame
        self.viewer.new_frame()

        # draw the world onto the frame
        self.world_view.draw_world_to_frame()

        # render the frame
        self.viewer.draw_frame()

    def pause_sim(self):
        GLib.source_remove(self.sim_event_source)
        self.viewer.control_panel_state_paused()

    def step_sim_once(self):
        self.pause_sim()
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
