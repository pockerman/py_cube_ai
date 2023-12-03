from typing import Any, TypeVar

from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from pycubeai.utils.mixins import WithQTableMixin
from pycubeai.utils.wrappers import time_func_wrapper
from pycubeai.utils.episode_info import EpisodeInfo
from pycubeai.utils import INFO

QTable = TypeVar('QTable')
Policy = TypeVar('Policy')
Env = TypeVar('Env')


class Sarsa(TDAlgoBase, WithQTableMixin):
    """SARSA algorithm

    """

    def __init__(self, algo_config: TDAlgoConfig):
        """Constructor

        Parameters
        ----------
        algo_config: Algorithm configuration
        """

        super().__init__(algo_config=algo_config)
        self.q_table = {}
        self.policy: Policy = algo_config.policy

    @property
    def q_function(self) -> QTable:
        return self.q_table

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """Any actions before the training begins

        Parameters
        ----------
        env: The training environment
        options: Any options passed by the client

        Returns
        -------
        None
        """
        super(Sarsa, self).actions_before_training_begins(env, **options)

        for state in range(env.n_states):
            for action in range(env.n_actions):
                self.q_table[state, action] = 0.0

    def actions_after_episode_ends(self, env: Env, episode_idx, **options) -> None:
        """Execute any actions the algorithm needs after the
        training episode ends

        Parameters
        ----------
        env: The training environment
        episode_idx: Training episode index
        options: Any options passed by the client

        Returns
        -------
        None
        """

        super(Sarsa, self).actions_after_episode_ends(env, episode_idx, **options)
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
        episode_score = 0.0

        # select an action
        action = self.policy(q_func=self.q_function, state=self.state)

        # dummy counter for how many iterations
        # we actually run. It is used to calculate the
        # average reward per episode
        counter = 0
        for itr in range(self.config.n_itrs_per_episode):

            # Take a step
            time_step = env.step(action)
            episode_score += time_step.reward

            next_state = time_step.observation
            next_action = self.policy(q_func=self.q_function, state=next_state)
            self.update_q_table(reward=time_step.reward, current_action=action, next_state=next_state, next_action=next_action)

            action = next_action
            self.state = next_state
            counter += 1

            if time_step.done:
                break

        episode_info = EpisodeInfo(episode_reward=episode_score, episode_index=episode_idx,
                                   episode_iterations=counter)
        return episode_info

    def update_q_table(self, reward: float, current_action: int, next_state: int, next_action: int) -> None:
        """Update the underlying q table

        Parameters
        ----------

        current_action: The action index selected by the policy
        reward: The reward returned by the environment
        next_state: The next state observed after taking the action
        next_action: The next action to take

        Returns
        -------

        None

        """

        # TD Update
        td_target = reward + self.gamma * self.q_function[next_state, next_action]
        td_delta = td_target - self.q_function[self.state, current_action]
        self.q_function[self.state, current_action] += self.alpha * td_delta






