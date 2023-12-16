"""module simple_grid_world. Models a three-cell grid.
This environment is taken from the book
Grokking Deep Reinforcement Learning by Miguel Morales, Manning Publications

"""

import enum
from pycubeai.worlds.time_step import TimeStep, StepType

class SimpleGridWorld(object):

    class Cell:
        def __init__(self, name: str, reward: int):
            self.name = name
            self.reward = reward

        def __eq__(self, other):

            if not isinstance(other, SimpleGridWorld.Cell):
                return False

            if self.name == other.name:
                return True

            return False

    class Action(enum.IntEnum):
        LEFT = -1
        RIGHT = 1

    VALID_ACTIONS = [Action.LEFT, Action.RIGHT]

    def __init__(self, discount: float = 0.1):
        self.discount = discount
        self.states = [SimpleGridWorld.Cell(name='H', reward=0),
                       SimpleGridWorld.Cell(name='S', reward=0),
                       SimpleGridWorld.Cell(name='G', reward=1)]
        self._current_state = self.states[1]


    def reset(self) -> TimeStep:
        self._current_state = self.states[1]
        return TimeStep(step_type=StepType.FIRST,
                        observation=self._current_state,
                        discount=self.discount,
                        reward=0,
                        info={})

    def step(self, action: Action) -> TimeStep:
        if action == SimpleGridWorld.Action.RIGHT:

            if self._current_state == self.states[0]:
                self._current_state = self.states[0]
                return TimeStep(step_type=StepType.MID,
                                observation=self._current_state,
                                reward=self._current_state.reward,
                                discount=self.discount,
                                info={})

            if self._current_state == self.states[1]:
                self._current_state = self.states[2]
                return TimeStep(step_type=StepType.LAST,
                                observation=self._current_state,
                                reward=self._current_state.reward,
                                discount=self.discount,
                                info={})
            # if we are in the hole we remain in the hole

            if self._current_state == self.states[2]:
                self._current_state = self.states[2]
                return TimeStep(step_type=StepType.MID,
                                observation=self._current_state,
                                discount=self.discount,
                                reward=0,
                                info={})

            raise ValueError("Invalid environment state for action")
        elif action == SimpleGridWorld.Action.LEFT:

            if self._current_state == self.states[0]:
                self._current_state = self.states[0]
                return TimeStep(step_type=StepType.MID,
                                observation=self._current_state,
                                reward=self._current_state.reward,
                                discount=self.discount,
                                info={})
            # we are at the start
            if self._current_state == self.states[1]:
                self._current_state = self.states[0]
                return TimeStep(step_type=StepType.MID,
                                observation=self._current_state,
                                reward=self._current_state.reward,
                                discount=self.discount,
                                info={})

            if self._current_state == self.states[2]:
                self._current_state = self.states[2]
                return TimeStep(step_type=StepType.MID,
                                observation=self._current_state,
                                discount=self.discount,
                                reward=0.0,
                                info={})

            raise ValueError("Invalid environment state for action")
