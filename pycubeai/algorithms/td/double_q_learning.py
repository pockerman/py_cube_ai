"""Module double_q_learning. Implements tabular double Q-learning
algorithm as presented in the paper

https://www.researchgate.net/publication/221619239_Double_Q-learning

"""

import numpy as np
from typing import TypeVar

from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from pycubeai.utils.mixins import WithDoubleMaxActionMixin
from pycubeai.utils.episode_info import EpisodeInfo
from pycubeai.worlds.time_step import TimeStep
from pycubeai.utils.wrappers import time_func_wrapper


Env = TypeVar('Env')
Policy = TypeVar('Policy')


class DoubleQLearning(TDAlgoBase, WithDoubleMaxActionMixin):
    """The class DoubleQLearning implements double q-learning tabular
    algorithm

    """

    def __init__(self, algo_config: TDAlgoConfig) -> None:
        """Constructor. Initialize the algorithm with the given configuration

        Parameters
        ----------
        algo_config: The algorithm configuration

        """

        super(DoubleQLearning, self).__init__(algo_config=algo_config)

        self.policy: Policy = algo_config.policy
        self.q1_table = {}
        self.q2_table = {}

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs before training begins

        Parameters
        ----------

        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        super(DoubleQLearning, self).actions_before_training_begins(env, **options)

        for state in range(env.n_states):
            for action in range(env.n_actions):
                self.q1_table[state, action] = 0.0
                self.q2_table[state, action] = 0.0

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options):
        """Execute any actions the algorithm needs after
        ending the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        super(DoubleQLearning, self).actions_after_episode_ends(env, episode_idx, **options)
        self.policy.actions_after_episode(episode_idx)

    @time_func_wrapper(show_time=False)
    def do_on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the agent on the environment at the given episode.

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Any options passes by the client code

        Returns
        -------

        An instance of the EpisodeInfo class

        """

        # episode score
        episode_score = 0
        counter = 0

        for itr in range(self.config.n_itrs_per_episode):

            # select an action
            action = self.policy(self.q1_table, self.q2_table, self.state)

            # take action A, observe R, S'
            time_step: TimeStep = env.step(action)
            episode_score += time_step.reward  # add reward to agent's score
            self._update_q_table(self.state, action, time_step.reward, time_step.observation)
            self.state = time_step.observation  # S <- S'

            if time_step.done:
                break

            counter += 1

        episode_info = EpisodeInfo(episode_reward=episode_score, episode_index=episode_idx, episode_iterations=counter)
        return episode_info

    def _update_q_table(self, env: Env, state: int, action: int, reward: float, next_state: int = None) -> None:
        """Update the Q-value function for the given state when taking the given action.
        The implementation chooses which of the two tables to update using a coin flip

        Parameters
        ----------

        env: The environment to train on
        state: The state currently on
        action: The action taken at the state
        reward: The reward taken
        next_state: The state to go when taking the given action

        Returns
        -------

        None
        """

        rand = np.random.random()

        if rand <= 0.5:

            # estimate in Q-table (for current state, action pair)
            q1_s = self.q1_table[state, action]

            max_act = self.one_table_max_action(self.q1_table, next_state,
                                                n_actions=env.action_space.n)

            # value of next state
            Qsa_next = \
                self.q2_table[next_state, max_act] if next_state is not None else 0

            # construct TD target
            target = reward + (self.gamma * Qsa_next)

            # get updated value
            new_value = q1_s + (self.alpha * (target - q1_s))
            self.q1_table[state, action] = new_value
        else:

            # estimate in Q-table (for current state, action pair)
            q2_s = self.q2_table[state, action]

            max_act = self.one_table_max_action(self.q2_table, next_state,
                                                n_actions=env.action_space.n)

            # value of next state
            Qsa_next = \
                self.q1_table[next_state, max_act] if next_state is not None else 0

            # construct TD target
            target = reward + (self.gamma * Qsa_next)

            # get updated value
            new_value = q2_s + (self.alpha * (target - q2_s))
            self.q2_table[state, action] = new_value






