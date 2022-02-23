"""
Tabular Q-learning algorithm
"""

from typing import Any, TypeVar
from src.algorithms.td.td_algorithm_base import TDAlgoBase, TDAlgoConfig
from src.utils.mixins import WithMaxActionMixin
from src.worlds.world_helpers import n_actions, n_states, step
from src.utils.episode_info import EpisodeInfo
from src.utils.time_step import TimeStep
from src.utils import INFO

Env = TypeVar('Env')
Policy = TypeVar('Policy')
QTable = TypeVar('QTable')


class QLearning(TDAlgoBase, WithMaxActionMixin):
    """
    epsilon-greedy Q-learning algorithm
    """

    def __init__(self, algo_config: TDAlgoConfig) -> None:
        """
        Constructor. Initialize by passing the configuration options

        Parameters
        ----------
        algo_config: The configuration options

        """
        super(QLearning, self).__init__(algo_config=algo_config)
        self.q_table = {}

    @property
    def q_function(self) -> QTable:
        return self.q_table

    @property
    def policy(self) -> Policy:
        return self.config.policy

    def on_state(self, state) -> Any:
        """Get an agent specific result e.g. an action or
        the state value

        Parameters
        ----------

        state The state instance presented to the agent

        Returns
        -------

        An agent specific result e.g. an action or
        the state value

        """
        return self.policy.select_action(self.q_table, state)

    def actions_before_training_begins(self, env: Env, **options) -> None:
        super(QLearning, self).actions_before_training_begins(env, **options)

        n_states_ = n_states(env)
        n_actions_ = n_actions(env)
        for state in range(n_states_):
            for action in range(n_actions_):
                self.q_table[state, action] = 0.0

    def actions_after_episode_ends(self, env: Env, episode_idx: int,  **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        super(QLearning, self).actions_after_episode_ends(env, episode_idx, **options)
        self.policy.actions_after_episode(episode_idx)

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """
        Perform one step of the algorithm
        """

        # episode score
        episode_score = 0
        counter = 0

        for itr in range(self.config.n_itrs_per_episode):

            if self.config.render_env:
                env.render()

            # epsilon-greedy action selection
            action = self.policy(q_func=self.q_function, state=self.state)

            # take action A, observe R, S'
            # next_state, reward, done, info = step(env, action)
            time_step: TimeStep = step(env, action)

            # add reward to agent's score
            episode_score += time_step.reward
            next_state = time_step.observation
            self._update_q_table(env=env, state=self.state, action=action,
                                 reward=time_step.reward, next_state=next_state)
            self.state = next_state  # S <- S'
            counter += 1

            if time_step.done:
                break

        #if episode_idx % self.output_msg_frequency == 0:
        #    print("{0}: On episode {1} training finished with  "
        #          "{2} iterations. Total reward={3}".format(INFO, current_episode_index, counter, episode_score))

        # self.iterations_per_episode.append(counter)
        # self.total_rewards[self.current_episode_index] = episode_score
        episode_info = EpisodeInfo(episode_reward=episode_score, episode_iterations=counter)
        return episode_info

    def _update_q_table(self, env: Env, state: int, action: int,
                        reward: float, next_state: int = None) -> None:
        """Update the underlying q table

        Parameters
        ----------
        env: The training environment
        state: The current state the environment in on
        action: The action index selected by the policy
        reward: The reward returned by the environment
        next_state: The next state observed after taking the action

        Returns
        -------

        None

        """

        # estimate in Q-table (for current state, action pair)
        q_s = self.q_function[state, action]

        # value of next state
        Qsa_next = \
            self.q_function[next_state, self.max_action(next_state,
                                                        n_actions=n_actions(env))] if next_state is not None else 0
        # construct TD target
        target = reward + (self.gamma * Qsa_next)

        # get updated value
        new_value = q_s + (self.alpha * (target - q_s))
        self.q_function[state, action] = new_value


