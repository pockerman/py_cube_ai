"""Module extended_kalman_filter. Implements
a simple extended Kalman filter

"""

import numpy as np
import copy
from typing import TypeVar
from dataclasses import dataclass

MotionModel = TypeVar('MotionModel')
ObservationModel = TypeVar('ObservationModel')


@dataclass(init=True, repr=True)
class EKFConfig(object):
    motion_model: MotionModel
    observation_model: ObservationModel
    q_matrix: np.array
    r_matrix: np.array
    l_matrix: np.array


class ExtendedKalmanFilter(object):

    def __init__(self, config: EKFConfig, init_state: np.array):
        self.config: EKFConfig = config
        self.sigma_points: np.ndarray = None
        self.state: np.array = copy.deepcopy(init_state)
        self.P = np.eye(init_state.shape[0])
        self.K: np.array = None

    def estimate(self, u: np.array, z: np.array) -> np.array:
        self.predict(u)
        self.coreect(z)

        return self.state, self.P

    def predict(self, u: np.array):

        # predict the state
        self.state = self.config.motion_model(self.state, u)

        # compute an new estimate for P
        F = self.config.motion_model.jacobian_matrix(self.state, u)
        self.P = F @ self.P @ F.T + self.config.l_matrix @ self.config.q_matrix @ self.config.l_matrix.T

    def coreect(self, z):

        jacobian = self.config.observation_model.jacobian_matrix()

        z_observation = self.config.observation_model(self.state)

        y = z - z_observation
        S = jacobian @ self.P @ jacobian.T + self.config.r_matrix
        self.K = self.P @ jacobian.T @ np.linalg.inv(S)

        # correct
        self.state = self.state + self.K @ y
        self.P = (np.eye(len(self.state)) - self.K @ jacobian) @ self.P





