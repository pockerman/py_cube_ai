from typing import Any
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GLib
#import gui.frame
#import gui.viewer

from src.simulator.simulator_base import SimulatorBase
from src.simulator.gui.viewer import Viewer
from src.simulator.worlds.default_world import DefaultWorld
from src.simulator.robots.diff_drive_robot import DiffDriveRobot
from src.utils import INFO

# hertz
REFRESH_RATE = 20.0

# meters
MAJOR_GRIDLINE_INTERVAL = 1.0

# minor gridlines for every major gridline
MAJOR_GRIDLINE_SUBDIVISIONS = 5


class WorldView(object):

    def __init__(self, world, viewer):
        self.world = world
        self.viewer = viewer

        # initialize views for world objects
        self.robot_views = []

        for robot in self.world.robots:
            self.add_robot_view(robot)

        self.obstacle_views = []
        for obstacle in self.world.obstacles:
            self.add_obstacle_view(obstacle)

    def set_viewer(self, viewer):
        self.viewer = viewer

    def draw_world_to_frame(self):
        # draw the grid
        self._draw_grid_to_frame()

        # draw all the robots
        for robot_view in self.robot_views:
            robot_view.draw_robot_to_frame()
        # draw all the obstacles
        for obstacle_view in self.obstacle_views:
            obstacle_view.draw_obstacle_to_frame()

    def _draw_grid_to_frame(self):
        # NOTE: THIS FORMULA ASSUMES THE FOLLOWING:
        # - Window size never changes
        # - Window is always centered at (0, 0)

        # calculate minor gridline interval
        minor_gridline_interval = MAJOR_GRIDLINE_INTERVAL / MAJOR_GRIDLINE_SUBDIVISIONS

        # determine world space to draw grid upon
        meters_per_pixel = 1.0 / self.viewer.pixels_per_meter
        width = meters_per_pixel * self.viewer.view_width_pixels
        height = meters_per_pixel * self.viewer.view_height_pixels
        x_halfwidth = width * 0.5
        y_halfwidth = height * 0.5

        x_max = int(x_halfwidth / minor_gridline_interval)
        y_max = int(y_halfwidth / minor_gridline_interval)

        # build the gridlines
        major_lines_accum = []  # accumulator for major gridlines
        minor_lines_accum = []  # accumulator for minor gridlines

        for i in range(x_max + 1):  # build the vertical gridlines
            x = i * minor_gridline_interval

            if x % MAJOR_GRIDLINE_INTERVAL == 0:  # sort major from minor
                accum = major_lines_accum
            else:
                accum = minor_lines_accum

            accum.append(
                [[x, -y_halfwidth], [x, y_halfwidth]]
            )  # positive-side gridline
            accum.append(
                [[-x, -y_halfwidth], [-x, y_halfwidth]]
            )  # negative-side gridline

        for j in range(y_max + 1):  # build the horizontal gridlines
            y = j * minor_gridline_interval

            if y % MAJOR_GRIDLINE_INTERVAL == 0:  # sort major from minor
                accum = major_lines_accum
            else:
                accum = minor_lines_accum

            accum.append(
                [[-x_halfwidth, y], [x_halfwidth, y]]
            )  # positive-side gridline
            accum.append(
                [[-x_halfwidth, -y], [x_halfwidth, -y]]
            )  # negative-side gridline

        # draw the gridlines
        self.viewer.current_frame.add_lines(
            major_lines_accum,  # draw major gridlines
            linewidth=meters_per_pixel,  # roughly 1 pixel
            color="black",
            alpha=0.2,
        )
        self.viewer.current_frame.add_lines(
            minor_lines_accum,  # draw minor gridlines
            linewidth=meters_per_pixel,  # roughly 1 pixel
            color="black",
            alpha=0.1,
        )

    def add_robot_view(self, robot):
        pass

    def add_obstacle_view(self, view):
        pass



class Simulator(SimulatorBase):

    def __init__(self, refresh_rate, world: Any, world_view: Any):
        super(Simulator, self).__init__(refresh_rate=refresh_rate, world=world, world_view=world_view)
        self.viewer = Viewer(simulator=self)
        self.world_view.set_viewer(viewer=self.viewer)

        # Gtk simulation event source - for simulation control
        # we use this opportunity to initialize the sim
        self.sim_event_source = GLib.idle_add(self.initialize_sim)

        # start Gtk

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

        self.draw_world()

    def draw_world(self):

        # start a fresh frame
        self.viewer.new_frame()

        # draw the world onto the frame
        self.world_view.draw_world_to_frame()

        # render the frame
        self.viewer.draw_frame()


if __name__ == '__main__':

    world = DefaultWorld(period=1.0/REFRESH_RATE)
    robot = DiffDriveRobot()

    world.add_robot(robot=robot)

    world_view = WorldView(world=world, viewer=None)

    simulator = Simulator(refresh_rate=REFRESH_RATE, world=world, world_view=world_view)

    simulator.simulate()

