"""Module episodic_sarsa_semi_gradient. Implements
semi-gradient SARSA for episodic environments

"""

from typing import TypeVar
import numpy as np
from dataclasses import dataclass

from src.algorithms.td. td_algorithm_base import TDAlgoBase, TDAlgoConfig
from src.utils.episode_info import EpisodeInfo


Env = TypeVar('Env')
Action = TypeVar('Action')
Policy = TypeVar('Policy')


@dataclass(init=True, repr=True)
class SemiGradSARSAConfig(TDAlgoConfig):

    dt_update_frequency: int = 100
    dt_update_factor: float = 1.0


class EpisodicSarsaSemiGrad(TDAlgoBase):
    """Episodic semi-gradient SARSA algorithm implementation
    """

    def __init__(self, algo_config: SemiGradSARSAConfig) -> None:
        """
        Constructor. Initialize the agent with the given configuration

        Parameters
        ----------

        algo_config: Configuration for the algorithm

        """
        super(EpisodicSarsaSemiGrad, self).__init__(algo_config=algo_config)

        # initialized in actions_before_training_begins
        self.weights: np.array = None
        self.dt = 1.0
        self.policy = algo_config.policy
        self.counters = {}

    def q_value(self, state_action: Action) -> float:
        return self.weights.dot(state_action)

    def update_weights(self, total_reward: float, state_action: Action,
                       state_action_: Action, t: float) -> None:
        """
        Update the weights
        Parameters
        ----------
        total_reward: The reward observed
        state_action: The action that led to the reward
        state_action_:
        t: The decay factor for alpha

        Returns
        -------

        None

        """
        v1 = self.q_value(state_action=state_action)
        v2 = self.q_value(state_action=state_action_)
        self.weights += self.alpha / t * (total_reward + self.gamma*v2 - v1) * state_action

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs before

        Parameters
        ----------
        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """

        self.weights: np.array = np.zeros((env.n_states * env.n_actions))

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs before
        starting the episode

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The episode index the algorithm trains on
        options: Any options passed by the client code

        Returns
        -------

        None

        """

        super(EpisodicSarsaSemiGrad, self).actions_before_episode_begins(env, episode_idx, **options)

        if episode_idx % self.config.dt_update_frequency == 0:
            self.dt += self.config.dt_update_factor

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Execute any actions the algorithm needs after
        ending the episode

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The episode index the algorithm trains on
        options: Any options passed by the client code

        Returns
        -------

        None

        """

        super(EpisodicSarsaSemiGrad, self).actions_after_episode_ends(env, episode_idx, **options)
        self.policy.actions_after_episode(episode_idx, **options)

    def select_action(self, env: Env, raw_state) -> int:

        # TODO: For epsilon greedy we may not have to calculate constantly
        vals = []
        for a in range(env.n_actions):
            sa = env.get_tiled_state(action=a, obs=raw_state)
            vals.append(self.q_value(state_action=sa))

        vals = np.array(vals)

        # choose an action at the current state
        action = self.policy(vals, raw_state)
        return action

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------
        env: The environment to run the training episode
        episode_idx: The episode index
        options: Options that client code may pass

        Returns
        -------

        An instance of EpisodeInfo

        """

        action = self.select_action(env, raw_state=self.state)
        count = 0
        episode_total_reward = 0.0

        for itr in range(self.config.n_itrs_per_episode):

            # this should change 
            state_action = env.get_tiled_state(action=action, obs=self.state)

            # step in the environment
            obs, reward, done, _ = env.step(action)
            episode_total_reward += reward

            if done and itr < env.get_property(prop_str="_max_episode_steps"):

                val = self.q_value(state_action=state_action)
                self.weights += self.alpha / self.dt * (reward - val) * state_action
                break

            new_action = self.select_action(raw_state=obs)
            sa = env.get_tiled_state(action=new_action, obs=obs)
            self.update_weights(total_reward=reward, state_action=state_action, state_action_=sa, t=self.dt)

            # update current state and action
            self.state = obs
            action = new_action
            count += 1

        self.counters[episode_idx] = count
        episode_info = EpisodeInfo()
        episode_info.episode_reward = episode_total_reward
        episode_info.episode_iterations = count
        return episode_info



