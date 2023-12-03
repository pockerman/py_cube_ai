"""
Learner for controlling a differential drive robot
"""
from typing import TypeVar
import json
from pathlib import Path
from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from pycubeai.utils.mixins import WithMaxActionMixin, WithQTableMixin
from pycubeai.utils import INFO
from pycubeai.py_cubeai_io.json_io import write_q_function, load_q_function

Env = TypeVar('Env')
Policy = TypeVar("Policy")
State = TypeVar('State')


class DiffDriveRobotQLearner(TDAlgoBase, WithQTableMixin, WithMaxActionMixin):

    def __init__(self, algo_in: TDAlgoConfig) -> None:
        super(DiffDriveRobotQLearner, self).__init__(algo_in=algo_in)
        self.q_table = {}
        self.policy: Policy = algo_in.policy
        self.training_finished: bool = False

    def save_q_function(self, filename: Path) -> None:
        write_q_function(qtable=self.q_table, filename=filename, **{})

    def load_q_function(self, filename: Path) -> None:
        table = load_q_function(filename=filename, **{})

        # need to transform the keys from strs
        # to a tuple
        for key in table:
            new_key = key.replace('(', '')
            new_key = new_key.replace(')', '')
            new_key_list = new_key.split(',')

            qtable_key = ((int(new_key_list[0]), int(new_key_list[1])), int(new_key_list[2]))
            self.q_table[qtable_key] = float(table[key])

    def play(self, env: Env, n_games: int):

        for game in range(n_games):
            print("{0} Playing game {1}".format(INFO, game))
            time_step = env.reset()
            counter = 0
            while env.continue_sim():
                action = self.get_action(state=env.resolve_time_step_as_key(time_step))
                action = env.get_action(action)

                print("{0} At state {1} action selected {2}".format(INFO, time_step.state.position, action.action_type.name))
                time_step = env.step(action)

                counter += 1

                #if counter >= 10:
                #    break

    def get_action(self, state: State):
        """
        Returns the action that should be followed when
        the algorithm is trained
        :param state:
        :return:
        """
        if not self.training_finished:
            raise ValueError("Training of the agent is not finished")

        return self.policy.select_action(self.q_table, state)

    def actions_before_training_begins(self, **options) -> None:
        """
        Any actions before training begins
        :param options:
        :return:
        """

        super(DiffDriveRobotQLearner, self).actions_before_training_begins(**options)

        for state in self.train_env.states:
            for action in self.train_env.actions:
                self.q_table[state, action] = 0.0

    def actions_after_training_ends(self, **options) -> None:
        super(DiffDriveRobotQLearner, self).actions_after_training_ends(**options)
        self.training_finished = True

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        super(DiffDriveRobotQLearner, self).actions_after_episode_ends()
        self.policy.actions_after_episode(self.current_episode_index)

    def on_episode(self, **options) -> None:
        """
        Perform one step of the algorithm
        """

        # episode score
        episode_score = 0
        counter = 0

        while self.train_env.continue_sim():


            # use policy to select an action
            action = self.policy(q_func=self.q_table, state=self.train_env.resolve_time_step_as_key(self.state))

            action = self.train_env.get_action(action)

            print("{0} INFO at state={1} action={2}".format(INFO, self.train_env.resolve_time_step_as_key(self.state), action.name))

            # take action A, observe R, S'
            time_step = self.train_env.step(action)

            # add reward to agent's score
            episode_score += time_step.reward
            self._update_Q_table(state=self.train_env.resolve_time_step_as_key(self.state),
                                 action=action.idx, reward=time_step.reward,
                                 next_state=self.train_env.resolve_time_step_as_key(time_step))

            # S <- S' in fact we update the whole time step
            self.state = time_step
            counter += 1

            if time_step.done or counter >= self.n_itrs_per_episode:
                break

        if self.current_episode_index % self.output_msg_frequency == 0:
            print("{0}: On episode {1} training finished with  "
                  "{2} iterations. Total reward={3}".format(INFO, self.current_episode_index, counter, episode_score))

        self.iterations_per_episode.append(counter)
        self.total_rewards[self.current_episode_index] = episode_score

    def _update_Q_table(self, state: State, action: int, reward: float, next_state: State = None) -> None:
        """
        Update the Q-value for the state
        """

        # estimate in Q-table (for current state, action pair)
        q_s = self.q_table[state, action]

        # value of next state
        Qsa_next = \
            self.q_table[next_state, self.max_action(self.q_table, next_state,
                                                        n_actions=self.train_env.n_actions)] if next_state is not None else 0
        # construct TD target
        target = reward + (self.gamma * Qsa_next)

        # get updated value
        new_value = q_s + (self.alpha * (target - q_s))
        self.q_table[state, action] = new_value


