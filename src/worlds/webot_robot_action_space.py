"""
Action space definition for Webots robots
"""
import abc
import enum
from abc import ABC


class WebotRobotActionType(enum.IntEnum):

    INVALID_TYPE = -1
    STOP = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    MOVE_FWD = 3
    MOVE_BWD = 4
    ROTATE = 5


class WebotRobotActionBase(ABC):
    """
    Base class for deriving Webot robot actions
    """
    def __init__(self, action_type: WebotRobotActionType):
        self.action_type = action_type
        self.idx: int = self.action_type

    @property
    def name(self) -> str:
        return self.action_type.name

    @abc.abstractmethod
    def act(self, *args, **options):
        """
        Perform the action
        :param args:
        :param options:
        :return:
        """

class WebotRobotMoveFWDAction(WebotRobotActionBase):
    """
    Move FWD action
    """

    def __init__(self, motor_speed):
        super(WebotRobotMoveFWDAction, self).__init__(action_type=WebotRobotActionType.MOVE_FWD)

        self.motor_speed = motor_speed

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(self.motor_speed)
        right_motor.setVelocity(self.motor_speed)

        #robot.step(TIME_STEP)


class WebotRobotMoveBWDAction(WebotRobotActionBase):

    def __init__(self, motor_speed):
        super(WebotRobotMoveBWDAction, self).__init__(action_type=WebotRobotActionType.MOVE_BWD)

        self.motor_speed = motor_speed

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(self.motor_speed)
        right_motor.setVelocity(self.motor_speed)

        #robot.step(TIME_STEP)


class WebotRobotStopAction(WebotRobotActionBase):

    def __init__(self):
        super(WebotRobotStopAction, self).__init__(action_type=WebotRobotActionType.STOP)

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        #robot.step(TIME_STEP)


class WebotRobotMoveTurnLeftAction(WebotRobotActionBase):

    def __init__(self, motor_speed):
        super(WebotRobotMoveTurnLeftAction, self).__init__(action_type=WebotRobotActionType.TURN_LEFT)

        self.motor_speed = motor_speed

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(self.motor_speed)

        #robot.step(TIME_STEP)


class WebotRobotMoveRightLeftAction(WebotRobotActionBase):

    def __init__(self, motor_speed):
        super(WebotRobotMoveRightLeftAction, self).__init__(action_type=WebotRobotActionType.TURN_RIGHT)

        self.motor_speed = motor_speed

    def act(self, *args, **options):

        robot = args[0]
        TIME_STEP = args[1]
        left_motor = robot.getDevice('left wheel motor')
        right_motor = robot.getDevice('right wheel motor')

        left_motor.setVelocity(self.motor_speed)
        right_motor.setVelocity(0.0)

        #robot.step(TIME_STEP)


class WebotRobotActionSpace(object):

    def __init__(self):
        self.actions = {}

    def add_action(self, action: WebotRobotActionBase) -> None:

        # update the id
        action.idx = len(self.actions)
        self.actions[action.idx] = action

    def __getitem__(self, item) -> WebotRobotActionBase:
        return self.actions[item]

    def __len__(self):
        return len(self.actions)


