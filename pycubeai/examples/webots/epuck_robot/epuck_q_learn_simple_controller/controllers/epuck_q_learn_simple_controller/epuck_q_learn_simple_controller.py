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
from pathlib import Path
import matplotlib.pyplot as plt

from controller import Supervisor, Node
from pycubeai.utils import INFO
from pycubeai.worlds.webot_environment_wrapper import EnvironmentWrapper, EnvConfig
from pycubeai.worlds.state_aggregation_webot_env import StateAggregationWebotEnv, StateAggregationWebotEnvBoundaries
from pycubeai.worlds.webot_robot_action_space import WebotRobotActionType, WebotRobotActionBase, WebotRobotMoveBWDAction, \
    WebotRobotMoveFWDAction, WebotRobotMoveRightLeftAction, WebotRobotStopAction, WebotRobotMoveTurnLeftAction

from pycubeai.agents.diff_drive_robot_qlearner import DiffDriveRobotQLearner
from pycubeai.algorithms.td.td_algorithm_base import TDAlgoConfig
from pycubeai.policies import EpsilonGreedyPolicy, EpsilonDecayOption

# Define a variable that defines the duration of each physics step.
# This macro will be used as argument to the Robot::step function,
# and it will also be used to enable the devices.
# This duration is specified in milliseconds and it must
# be a multiple of the value in the basicTimeStep field of the WorldInfo node.
TIME_STEP = 32

MAX_SPEED = 6.28
MIN_SPEED = 0.0

# threshold value for the proximity sensors
# to identify the fact that the robot crushed the wall
BUMP_THESHOLD = 90
N_EPISODES = 1000
PLOT_STEP = 10
N_ITRS_PER_EPISODE = 2000
EPS = 1.0
EPS_DECAY_OP = EpsilonDecayOption.INVERSE_STEP


def plot_running_avg(avg_rewards, step):

    running_avg = np.empty(avg_rewards.shape[0])
    for t in range(avg_rewards.shape[0]):
        running_avg[t] = np.mean(avg_rewards[max(0, t - step): (t + 1)])
    plt.plot(running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Reward")
    plt.title("Running average")
    plt.show()


class Policy(EpsilonGreedyPolicy):
    def __init__(self, n_actions: int) -> None:
        super(Policy, self).__init__(eps=EPS, decay_op=EPS_DECAY_OP, n_actions=n_actions)

    def __call__(self, q_func, state) -> int:

        # if we are at the origin always choose FWD
        if state == (5, 5):
            return 1

        # if we are close to the goal
        # then stop
        #if state == (1, 5) or state == (5, 1):
        #    return 0

        return super(Policy, self).__call__(q_func, state)

    def select_action(self, q_func, state) -> int:
        """
        Deterministically choose the best action from
        the given q-table at the given state
        :param q_func:
        :param state:
        :return:
        """

        # if we are at the origin always choose FWD
        if state == (5, 5):
            return 1

        return super(Policy, self).max_action(q_func, state=state, n_actions=self.n_actions)


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
        # we want to make progress. If we are at the
        # start position and decide to STOP then exit the game
        if l2_norm < 1.0e-4 and action.action_type == WebotRobotActionType.STOP:
            return True, -2.0, l2_norm

        # compute l2 norm from goal
        l2_norm = np.linalg.norm(position - self.goal_position)

        if l2_norm < self.goal_radius:

            # we reached the goal but we also want
            # the robot to stop

            if action.action_type == WebotRobotActionType.STOP:
                return True, 10.0, l2_norm
            else:
                # otherwise punish the agent
                return False, -2.0, l2_norm

        # goal has not been reached. No reason to stop
        # so penalize this choice
        if action.action_type == WebotRobotActionType.STOP:
            return False, -2.0, l2_norm

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
    env_config.dt = TIME_STEP
    env_config.robot_name = "qlearn_e_puck"
    env_config.bump_threshold = BUMP_THESHOLD
    env_config.on_goal_criterion = on_goal_criterion
    env_config.reward_on_wall_crush = -5.0
    environment = EnvironmentWrapper(robot=robot, robot_node=robot_node, config=env_config)

    environment.add_action(action=WebotRobotStopAction())
    environment.add_action(action=WebotRobotMoveFWDAction(motor_speed=MAX_SPEED))

    # position aggregation environment
    boundaries = StateAggregationWebotEnvBoundaries(xcoords=(-3.0, 3.0),
                                                    ycoords=(-3.0, 3.0))

    state_aggregation_env = StateAggregationWebotEnv(env=environment,
                                                     boundaries=boundaries, states=(10, 10))

    agent_config = TDAlgoConfig()
    agent_config.n_episodes = N_EPISODES
    agent_config.n_itrs_per_episode = N_ITRS_PER_EPISODE
    agent_config.gamma = 0.99
    agent_config.alpha = 0.1
    agent_config.output_freq = 1
    agent_config.train_env = state_aggregation_env
    agent_config.policy = Policy(n_actions=state_aggregation_env.n_actions)

    agent = DiffDriveRobotQLearner(algo_in=agent_config)
    agent.train()

    agent.save_q_function(filename=Path("/home/alex/qi3/rl_python/pycubeai/apps/webots/diff_drive_sys/controllers/epuck_q_learn_simple_controller/q_learner.json"))
    plot_running_avg(agent.total_rewards, step=PLOT_STEP)

    #print(agent.q_table)
    #print("{0} Finished training".format(INFO))
    # once the agent is trained let's play
    agent.training_finished = True
    agent.load_q_function(filename=Path("/home/alex/qi3/rl_python/pycubeai/apps/webots/diff_drive_sys/controllers/epuck_q_learn_simple_controller/q_learner.json"))
    agent.play(env=state_aggregation_env, n_games=1)


# Enter here exit cleanup code.
if __name__ == '__main__':

    controller_main()

