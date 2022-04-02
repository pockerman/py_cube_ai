import numpy as np
import math
from src.filtering.extended_kalman_filter import ExtendedKalmanFilter, EKFConfig

DT = 0.1
SIM_TIME = 50.0


class MotionModel(object):


    def __init__(self):
        pass

    def __call__(self, x, u) -> np.array:
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[DT * math.cos(x[2, 0]), 0],
                      [DT * math.sin(x[2, 0]), 0],
                      [0.0, DT],
                      [1.0, 0.0]])

        return F @ x + B @ u


class ObservationModel(object):
    def __init__(self):
        pass


    def __call__(self, x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H @ x
        return z



if __name__ == '__main__':
    # Covariance for EKF simulation
    Q = np.diag([
        0.1,  # variance of location on x-axis
        0.1,  # variance of location on y-axis
        np.deg2rad(1.0),  # variance of yaw angle
        1.0  # variance of velocity
    ]) ** 2  # predict state covariance

    # Observation x,y position covariance
    R = np.diag([1.0, 1.0]) ** 2

    ekf_config = EKFConfig(q_matrix=Q, r_matrix=R)
