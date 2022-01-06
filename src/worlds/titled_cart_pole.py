"""
CartPole environment with state aggregation.
 
"""

import numpy as np
import gym
from typing import TypeVar, Any
from src.worlds.tiled_world_wrapper import TiledEnvWrapper
from src.utils.exceptions import InvalidParameterValue

Env = TypeVar('Env')
State = TypeVar('State')
Action = TypeVar('Action')
TiledState = TypeVar('TiledState')


class TiledCartPole(TiledEnvWrapper):
    
    def __init__(self, n_states: int, state_var_idx: int = 2, n_actions: int = 2, version: str = "v0") -> None:
        super(TiledCartPole, self).__init__(env=gym.make("CartPole-" + version),
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
        self.create_bins()

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
            self._build_theta_velocity_space()
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
        self.pole_theta_space = np.linspace(-0.20943951, 0.20943951, self.n_states)

    def _build_theta_velocity_space(self) -> None:
        self.pole_theta_velocity_space = np.linspace(-4, 4, self.n_states)

    def _build_cart_position_space(self) -> None:
        self.cart_pos_space = np.linspace(-2.4, 2.4, self.n_states)

    def _build_cart_velocity_space(self) -> None:
        self.cart_vel_space = np.linspace(-4, 4, self.n_states)

