
from abc import abstractmethod
import numpy as np


class PolicyEstimatorBase(object):

    """
    Base class for policy estimators
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_action_probabilities(self, state) -> np.array:
        pass

    @abstractmethod
    def update(self, state, total_return) -> None:
        pass