from src.simulator.simulators.default_simulator import DefaultSimulator
from src.simulator.gui.viewer import Viewer
from src.simulator.viewers.world_view import WorldView
from src.simulator.maps.map_manager import MapManager
from src.simulator.worlds.default_world import DefaultWorld
from src.simulator.robots.kherera_iii import KheperaIII
from src.utils import INFO

# hertz
REFRESH_RATE = 20.0


if __name__ == '__main__':

    # the robot to simulate
    robot = KheperaIII()

    viewer = Viewer(simulator=None)
    world = DefaultWorld(period=1.0/REFRESH_RATE)

    world.add_robot(robot=robot)

    world_view = WorldView(world=world, viewer=viewer)

    print("{0} Number of robots in world {1}".format(INFO, len(world.robots)))

    map_manager = MapManager()

    simulator = DefaultSimulator(refresh_rate=REFRESH_RATE, viewer=viewer,
                                 world=world, world_view=world_view, map_manager=map_manager)

    simulator.simulate()

