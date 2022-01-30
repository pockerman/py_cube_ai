"""
RL Environment API taken from
https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py
"""


import enum

from typing import NamedTuple, Generic, Optional, TypeVar


_Reward = TypeVar('_Reward')
_Discount = TypeVar('_Discount')
_Observation = TypeVar('_Observation')
_INFO = TypeVar('_INFO')


class StepType(enum.IntEnum):
    """
    Defines the status of a `TimeStep` within a sequence.
    """

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0

    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1

    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST


class TimeStep(NamedTuple, Generic[_Reward, _Discount, _Observation, _INFO]):
    """
    The TimeStep class
    """
    step_type: StepType
    reward: Optional[_Reward]
    discount: Optional[_Discount]
    observation: _Observation
    info: Optional[_INFO]

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST
