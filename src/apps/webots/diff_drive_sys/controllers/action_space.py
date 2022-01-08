import abc

import gym
import enum
from abc import abstractmethod, ABC


class ActionType(enum.IntEnum):

    STOP = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    MOVE_FWD = 3
    MOVE_BWD = 4
    ROTATE = 5


class ActionBase(ABC):
    """

    """
    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abc.abstractmethod
    def act(self, *args, **options):
        """
        
        :param args:
        :param options:
        :return:
        """


class MoveFWDAction(ActionBase):

    def __init__(self):
        super(MoveFWDAction, self).__init__(action_type=ActionType.MOVE_FWD)

        self.left_motor_speed = 0
        self.right_motor_speed = 0

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(self.left_motor_speed)
        right_motor.setVelocity(self.right_motor_speed)

        robot.step(TIME_STEP)


