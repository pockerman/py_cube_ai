from abc import abstractmethod
import numpy as np


class ValueEstimatorBase(object):

    """
    Base class for policy estimators
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_baseline(self, state) -> float:
        pass

    @abstractmethod
    def update(self, state, advantage, action) -> None:
        pass