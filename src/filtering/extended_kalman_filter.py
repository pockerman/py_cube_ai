"""Module extended_kalman_filter. Implements
a simple extended Kalman filter

"""

import numpy as np
from typing import TypeVar
from dataclasses import dataclass

MotionModel = TypeVar('MotionModel')
ObservationModel = TypeVar('ObservationModel')


@dataclass(init=True, repr=True)
class EKFConfig(object):
    motion_model: MotionModel
    observation_model: ObservationModel
    q_matrix: np.ndarray
    r_matrix: np.ndarray


class ExtendedKalmanFilter(object):

    def __init__(self, config: EKFConfig):
        self.config: EKFConfig = config
        self.sigma_points: np.ndarray = None
        self.old_state = None

    def predict(self, u: np.ndarray):

        # predict the state
        state = self.config.motion_model(self.old_state, u)


    def _init_sigma_points(self):
        pass


