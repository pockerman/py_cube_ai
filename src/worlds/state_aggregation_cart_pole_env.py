"""
CartPole environment with state aggregation. The original environment
is described here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
The state variables are:

Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -2.4                    2.4
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.209 rad (-12 deg)    0.209 rad (12 deg)
        3       Pole Angular Velocity     -Inf                    Inf

However, in the implementation below, the default behavior is to constrain these variables
"""

import numpy as np
import gym
from typing import TypeVar
from collections import namedtuple
from src.worlds.state_aggregator_world_wrapper import StateAggregationEnvWrapper
from src.utils.exceptions import InvalidParameterValue

Env = TypeVar('Env')
State = TypeVar('State')
Action = TypeVar('Action')
TiledState = TypeVar('TiledState')

# boundaries for the variables in CartPole environment
StateAggregationCartPoleBounds = namedtuple('TiledCartPoleBounds', ['cart_position_space', 'cart_velocity_space',
                                                         'pole_theta_space', 'pole_theta_velocity_space'])


class StateAggregationCartPoleEnv(StateAggregationEnvWrapper):
    
    def __init__(self, n_states: int, state_var_idx: int = 2,
                 n_actions: int = 2, version: str = "v0",
                 boundaries: StateAggregationCartPoleBounds = StateAggregationCartPoleBounds((-2.4, 2.4), (-4, 4), (-0.20943951, 0.20943951), (-4, 4))) -> None:
        super(StateAggregationCartPoleEnv, self).__init__(env=gym.make("CartPole-" + version),
                                                          n_actions=n_actions, n_states=n_states)

        if 0 > state_var_idx > 4:
            raise InvalidParameterValue("state_var_idx", param_val=[0, 3])

        # bins for the pole position space
        self.pole_theta_space = []

        # bins for the pole velocity space
        self.pole_theta_velocity_space = []

        # bins for the cart position space
        self.cart_pos_space = []

        # bins for the cart velocity space
        self.cart_vel_space = []

        self.state_var_idx = state_var_idx
        self.boundaries: StateAggregationCartPoleBounds = boundaries
        self.create_bins()
        self.create_state_space()

    def reset(self) -> State:
        """
        Reset the underlying environment
        :return: State
        """
        return self.get_state_from_obs(self.raw_env.reset())

    def step(self, action: Action):
        """
        Step in the environment
        :param action:
        :return:
        """
        obs, reward, done, info = self.raw_env.step(action=action)
        return self.get_state_from_obs(obs), reward, done, info

    def get_tiled_state(self, obs: State, action: Action) -> TiledState:
        """
        Returns the tiled states for the given observation
        :param obs: The observation to be tiled
        :param action: The action corresponding to the states
        :return: TiledState
        """

        cart_x, cart_x_dot, cart_theta, cart_theta_dot = obs

        if self.state_var_idx == 4:

            cart_x = int(np.digitize(cart_x, self.cart_pos_space))
            cart_x_dot = int(np.digitize(cart_x_dot, self.cart_vel_space))
            cart_theta = int(np.digitize(cart_theta, self.pole_theta_space))
            cart_theta_dot = int(np.digitize(cart_theta_dot, self.pole_theta_velocity_space))
            return cart_x, cart_x_dot, cart_theta, cart_theta_dot
        elif self.state_var_idx == 0:
            return int(np.digitize(cart_x, self.cart_pos_space))
        elif self.state_var_idx == 1:
            return int(np.digitize(cart_x_dot, self.cart_vel_space))
        elif self.state_var_idx == 2:
            return int(np.digitize(cart_theta, self.pole_theta_space))
        elif self.state_var_idx == 3:
            return int(np.digitize(cart_theta_dot, self.pole_theta_velocity_space))

    def create_bins(self) -> None:
        """
        Create the bins that the state variables of
        the underlying environment will be distributed
        :return: A list of bins for every state variable
        """

        if self.state_var_idx == 4:
            self._build_theta_velocity_space()
            self._build_pole_theta_space()
            self._build_cart_position_space()
            self._build_cart_velocity_space()
        elif self.state_var_idx == 2:
            self._build_pole_theta_space()
        elif self.state_var_idx == 0:
            self._build_cart_position_space()
        elif self.state_var_idx == 1:
            self._build_cart_velocity_space()
        elif self.state_var_idx == 3:
            self._build_theta_velocity_space()

    def create_state_space(self) -> None:
        """
        Creates the discrete state space.
        :return:
        """

        if self.state_var_idx == 4:
            for i in range(len(self.cart_pos_space) + 1):
                for j in range(len(self.cart_vel_space) + 1):
                    for k in range(len(self.pole_theta_space) + 1):
                        for l in range(len(self.pole_theta_velocity_space) + 1):
                            self.discrete_observation_space.append((i, j, k, l))
        else:
            for i in range(self.n_states + 1):
                self.discrete_observation_space.append(i)

    def _build_pole_theta_space(self) -> None:
        low = self.boundaries.pole_theta_space[0]
        high = self.boundaries.pole_theta_space[1]
        self.pole_theta_space = np.linspace(low, high, self.n_states)

    def _build_theta_velocity_space(self) -> None:
        low = self.boundaries.pole_theta_velocity_space[0]
        high = self.boundaries.pole_theta_velocity_space[1]
        self.pole_theta_velocity_space = np.linspace(low, high, self.n_states)

    def _build_cart_position_space(self) -> None:
        low = self.boundaries.cart_position_space[0]
        high = self.boundaries.cart_position_space[1]
        self.cart_pos_space = np.linspace(low, high, self.n_states)

    def _build_cart_velocity_space(self) -> None:
        low = self.boundaries.cart_velocity_space[0]
        high = self.boundaries.cart_velocity_space[1]
        self.cart_vel_space = np.linspace(low, high, self.n_states)

