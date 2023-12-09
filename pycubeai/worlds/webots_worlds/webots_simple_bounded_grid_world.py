"""Module simple_bounded_grid_world. Wrapper
to the Webots simple world

"""

from typing import TypeVar
from dataclasses import dataclass

from pycubeai.worlds.time_step import TimeStep

Supervisor = TypeVar('Supervisor')
Robot = TypeVar('Robot')
Action = TypeVar('Action')
RewardManager = TypeVar('RewardManager')
GoalCriterion = TypeVar('GoalCriterion')


@dataclass(init=True, repr=True)
class SimpleBoundedGridWorldConfiguration(object):

    supervisor: Supervisor = None
    robot: Robot = None
    dt: int = -1
    bump_threshold = 0
    on_goal_criterion: GoalCriterion = None
    reward_manager: RewardManager = None


class SimpleBoundedGridWorld(object):

    def __init__(self, configuration: SimpleBoundedGridWorldConfiguration):
        self.config = configuration

    def reset(self) -> TimeStep:
        pass

    def step(self, action) -> TimeStep:
        pass