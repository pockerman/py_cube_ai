"""
State aggregation version of the webots environment
"""

import numpy as np
from collections import namedtuple
from pycubeai.worlds.webot_environment_wrapper import EnvironmentWrapper, State
from pycubeai.worlds.webot_robot_action_space import WebotRobotActionBase
from pycubeai.worlds.time_step import TimeStep


StateAggregationWebotEnvBoundaries = namedtuple('StateAggregationWebotEnvBoundaries',
                                                ['xcoords', 'ycoords'])


class StateAggregationWebotEnv(object):
    """
    Wrapper that uses state aggregation for Webot environment
    """

    def __init__(self, env: EnvironmentWrapper,
                 boundaries: StateAggregationWebotEnvBoundaries, states: tuple) ->None:

        self.raw_env = env
        self.boundaries = boundaries
        self.n_x_states = states[0]
        self.n_y_states = states[1]
        self.n_states = self.n_x_states * self.n_y_states

        # list holding the bins for the xdim
        self.xdim_bins = []

        # list holding the bins for the xdim
        self.ydim_bins = []

        # the discrete observation space
        self.discrete_observation_space = []

        self._create_bins()
        self._create_state_space()

    @property
    def states(self):
        return self.discrete_observation_space

    @property
    def actions(self):
        return self.raw_env.actions

    @property
    def n_actions(self) -> int:
        return self.raw_env.n_actions

    def continue_sim(self) -> bool:
        """
        Returns true if the simulation is continued.
        We need to call robot step to synchronize the
        robot sensors etc
        :return:
        """
        return self.raw_env.continue_sim()

    def get_action(self, aidx) -> WebotRobotActionBase:
        return self.raw_env.get_action(aidx)

    def resolve_time_step_as_key(self, time_step: TimeStep) -> tuple:

        if isinstance(time_step, TimeStep):
            return time_step.state.position

        if isinstance(time_step, tuple):
            return time_step

        raise ValueError("type(time_step)= {0} cannot be resolved as tuple".format(type(time_step)))

    def reset(self) -> TimeStep:
        """
        Reset the underlying environment
        :return: State
        """

        # get the raw observation from the environment
        time_step = self.raw_env.reset()
        time_step = self._sanitize_time_step(time_step=time_step)
        return time_step

    def step(self, action: WebotRobotActionBase) -> TimeStep:
        """
        Execute the action on the simulator
        :param action:
        :return:
        """
        time_step = self.raw_env.step(action=action)
        time_step = self._sanitize_time_step(time_step=time_step)
        return time_step

    def get_position_bins(self, raw_position) -> tuple:

        xcoord = raw_position[0]
        ycoord = raw_position[1]

        xcoord = int(np.digitize(xcoord, self.xdim_bins))
        ycoord = int(np.digitize(ycoord, self.ydim_bins))

        return xcoord, ycoord

    def _sanitize_time_step(self, time_step) -> TimeStep:
        """
        Pass the necessary variables from the bins
        :param time_step:
        :return:
        """

        position = time_step.state.position

        position_bin = self.get_position_bins(raw_position=position[1:])

        state = State(position=position_bin, velocity=time_step.state.velocity,
                      orientation=time_step.state.orientation,
                      sensors=time_step.state.sensors,
                      motors=time_step.state.motors)

        time_step = TimeStep(state=state, done=time_step.done, reward=time_step.reward, info=time_step.info)
        return time_step

    def _create_state_space(self) -> None:
        """
        Creates the discrete state space.
        :return:
        """

        for i in range(len(self.xdim_bins) + 1):
            for j in range(len(self.ydim_bins) + 1):
                self.discrete_observation_space.append((i, j))

    def _create_bins(self) -> None:
        """
        Create the bins that the state variables of
        the underlying environment will be distributed
        :return: A list of bins for every state variable
        """

        # create the bins for x direction
        low = self.boundaries.xcoords[0]
        high = self.boundaries.xcoords[1]
        self.xdim_bins = np.linspace(low, high, self.n_x_states)

        # create the bins for the y direction
        low = self.boundaries.ycoords[0]
        high = self.boundaries.ycoords[1]
        self.ydim_bins = np.linspace(low, high, self.n_y_states)



