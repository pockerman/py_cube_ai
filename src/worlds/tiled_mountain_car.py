"""
MountainCar environment with state aggregation. The original environment
is described here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
The state variables are:

Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

The action space is:

Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
"""

import numpy as np
import gym
from collections import namedtuple
from typing import TypeVar, Any
from src.worlds.tiled_world_wrapper import TiledEnvWrapper
from src.utils.exceptions import InvalidParameterValue

State = TypeVar('State')
Action = TypeVar('Action')
TiledState = TypeVar('TiledState')

# boundaries for the variables in MountainCar environment
TiledMountainCarBounds = namedtuple('TiledMountainCarBounds', ['car_position_space', 'car_velocity_space'])


class TiledMountainCarEnv(TiledEnvWrapper):
    
    def __init__(self, version: str, n_states: int, state_var_idx: int = 2,
                 state_bounds: TiledMountainCarBounds = TiledMountainCarBounds(car_position_space=(-1.2, 0.6),
                                                                               car_velocity_space=(-0.07, 0.07))):
        super(TiledMountainCarEnv, self).__init__(env=gym.make("MountainCar-"+version),
                                                  n_states=n_states, n_actions=3)

        if state_var_idx != 0 and state_var_idx != 1 and state_var_idx != 2:
            raise InvalidParameterValue("state_var_idx", param_val=[0, 1, 2])

        self.state_var_idx = state_var_idx
        self.state_boundaries = state_bounds

        # bins for the pole position space
        self.car_position_space = []

        # bins for the pole velocity space
        self.car_velocity_space = []

        self.create_bins()
        self.create_state_space()

    @property
    def action_space(self) -> Any:
        return self.raw_env.action_space

    def reset(self) -> State:
        """
        Reset the underlying environment
        :return: State
        """
        obs = self.raw_env.reset()
        return self.get_state_from_obs(obs=obs)

    def get_property(self, prop_str: str) -> Any:
        """
        Get the property in the underlying environment
        :param prop_str:
        :return:
        """
        return self.raw_env.__dict__[prop_str]

    def step(self, action: Action):
        """
        Step in the environment
        :param action:
        :return:
        """
        obs, reward, done, info = self.raw_env.step(action=action)
        return self.get_state_from_obs(obs=obs), reward, done, info

    def get_state_from_obs(self, obs: State) -> TiledState:
        """
        Returns the state idx given the raw observation
        :param obs: Raw observation
        :return: tiled state
        """
        return self.get_tiled_state(obs=obs, action=None)

    def get_tiled_state(self, obs: State, action: Action) -> TiledState:
        """
        Returns the tiled states for the given observation
        :param obs: The observation to be tiled
        :param action: The action corresponding to the states
        :return: TiledState
        """

        car_position, car_velocity = obs

        if self.state_var_idx == 2:

            car_position = int(np.digitize(car_position, self.car_position_space))
            car_velocity = int(np.digitize(car_velocity, self.car_velocity_space))
            return car_position, car_velocity
        elif self.state_var_idx == 0:
            return int(np.digitize(car_position, self.car_position_space))
        elif self.state_var_idx == 1:
            return int(np.digitize(car_velocity, self.car_velocity_space))

    def create_bins(self) -> None:
        """
        Create the bins that the state variables of
        the underlying environment will be distributed
        :return: A list of bins for every state variable
        """
        if self.state_var_idx == 2:
            self._build_car_position_space()
            self._build_car_velocity_space()
        elif self.state_var_idx == 0:
            self._build_car_position_space()
        elif self.state_var_idx == 1:
            self._build_car_velocity_space()

    def create_state_space(self) -> None:
        """
        Creates the discrete state space.
        :return:
        """
        if self.state_var_idx == 2:
            for i in range(len(self.car_position_space) + 1):
                for j in range(len(self.car_velocity_space) + 1):
                            self.discrete_observation_space.append((i, j))
        else:
            for i in range(self.n_states + 1):
                self.discrete_observation_space.append(i)

    def _build_car_position_space(self) -> None:
        low = self.state_boundaries.car_position_space[0]
        high = self.state_boundaries.car_position_space[1]
        self.car_position_space = np.linspace(low, high, self.n_states)

    def _build_car_velocity_space(self) -> None:
        low = self.state_boundaries.car_velocity_space[0]
        high = self.state_boundaries.car_velocity_space[1]
        self.car_velocity_space = np.linspace(low, high, self.n_states)

