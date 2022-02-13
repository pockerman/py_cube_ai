"""epuck_q_learn_simple_controller controller.
See the discussion here about the formula we use
to calculate the distance from the wall:

https://photo.stackexchange.com/questions/12434/how-do-i-calculate-the-distance-of-an-object-in-a-photo

The physical data for the model can be found here

https://cyberbotics.com/doc/guide/epuck#e-puck-model

Simulation of a differential drive system. In this simulation, we train a Q-learning
agent using the distance from the distance sensor. The environment is an empty
environment surrounded by a wall. The agent is located in the middle of he world
Its goal is to move around the perimeter of the environment without bumping on the wall.
Every time the robot bumps on the wall, the episode finishes and the agent receives a reward
of -1. Every time the agent completes a round it receives  a reward of 1.
If the agent is trapped in a corner.

The description of the epuck robot can be found at: https://cyberbotics.com/doc/guide/epuck#e-puck-model
"""

import numpy as np
from collections import namedtuple

from controller import Supervisor, Node
from src.utils import INFO
from src.worlds.webot_environment_wrapper import EnvironmentWrapper, EnvConfig
from src.worlds.state_aggregation_webot_env import StateAggregationWebotEnv, StateAggregationWebotEnvBoundaries
from src.worlds.webot_robot_action_space import WebotRobotActionType, WebotRobotActionBase, WebotRobotMoveBWDAction, \
    WebotRobotMoveFWDAction, WebotRobotMoveRightLeftAction, WebotRobotStopAction, WebotRobotMoveTurnLeftAction

from src.agents.diff_drive_robot_qlearner import DiffDriveRobotQLearner
from src.algorithms.td.td_algorithm_base import TDAlgoInput
from src.policies import EpsilonGreedyPolicy, EpsilonDecayOption

# Define a variable that defines the duration of each physics step.
# This macro will be used as argument to the Robot::step function,
# and it will also be used to enable the devices.
# This duration is specified in milliseconds and it must
# be a multiple of the value in the basicTimeStep field of the WorldInfo node.

TIME_STEP = 64
SPEED_RATE_FACTOR = 0.1
WALL_REAL_HEIGHT = 0.1

MAX_SPEED = 6.28
MIN_SPEED = 0.0

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 90

State = namedtuple("State", ["sensors", "motors"])

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)


class OnGoal(object):

    def __init__(self, goal_position: list) -> None:

        # radius away from the goal
        self.goal_radius: float = 0.1
        self.robot_radius = 7.4 / 100.0
        self.goal_position = np.array(goal_position)
        self.start_position = np.array([0., 0., 0., ])

    def check(self, robot_node: Node, action: WebotRobotActionBase) -> tuple:

        position = robot_node.getPosition()
        position = np.array(position)

        # compute l2 norm from goal
        l2_norm = np.linalg.norm(position - self.start_position)

        # we don't want to be stacked where we started
        # we want to make progress
        if l2_norm < 1.0e-4 and action.action_type == WebotRobotActionType.STOP:
            return False, -2.0, l2_norm

        # compute l2 norm from goal
        l2_norm = np.linalg.norm(position - self.goal_position)

        if l2_norm < self.goal_radius:

            # we reached the goal but we also want
            # the robot to stop

            if action.action_type == WebotRobotActionType.STOP:
                return True, 10.0, l2_norm
            else:
                return False, 5.0, l2_norm

        return False, 0.0, l2_norm


def controller_main():
    """
    Create the robot, establish the Environment. Train the QLearner
    :return:
    """

    # number of steps to play
    supervisor = Supervisor()

    robot_node = supervisor.getFromDef(name='qlearn_e_puck')

    if robot_node is None:
        raise ValueError("Robot node is None")

    robot_node.enablePoseTracking(TIME_STEP)

    goal_position = [0.0, 0.0, -2.5]
    on_goal_criterion = OnGoal(goal_position=goal_position)

    robot = supervisor
    env_config = EnvConfig()
    env_config.dt = 32
    env_config.robot_name = "qlearn_e_puck"
    env_config.bump_threshold = BUMP_THESHOLD
    env_config.on_goal_criterion = on_goal_criterion
    env_config.reward_on_wall_crush = -5.0
    environment = EnvironmentWrapper(robot=robot, robot_node=robot_node, config=env_config)

    environment.add_action(action=WebotRobotStopAction())
    environment.add_action(action=WebotRobotMoveFWDAction(motor_speed=0.5*MAX_SPEED))

    # position aggregation environment
    boundaries = StateAggregationWebotEnvBoundaries(xcoords=(-3.0, 3.0),
                                                    ycoords=(-3.0, 3.0))

    state_aggregation_env = StateAggregationWebotEnv(env=environment,
                                                     boundaries=boundaries, states=(10, 10))

    agent_config = TDAlgoInput()
    agent_config.n_episodes = 20
    agent_config.n_itrs_per_episode = 3000
    agent_config.gamma = 0.99
    agent_config.alpha = 0.1
    agent_config.output_freq = 1
    agent_config.train_env = state_aggregation_env
    agent_config.policy = EpsilonGreedyPolicy(eps=1.0, decay_op=EpsilonDecayOption.INVERSE_STEP, n_actions=state_aggregation_env.n_actions)

    agent = DiffDriveRobotQLearner(algo_in=agent_config)
    agent.train()

    print("{0} Finished training")
    # once the agent is trained let's play
    agent.play(env=state_aggregation_env, n_games=2)


# Enter here exit cleanup code.
if __name__ == '__main__':

    controller_main()

