"""The module dummy_gym_agent.
Specifies a dummy agent

"""
from typing import TypeVar, Any
from dataclasses import dataclass

from src.algorithms.rl_agent_base import RLAgentBase
from src.utils.episode_info import EpisodeInfo
from src.utils.play_info import PlayInfo

Env = TypeVar('Env')
Criterion = TypeVar('Criterion')
State = TypeVar('State')


@dataclass(init=True, repr=True)
class DummyAlgoConfig(object):

    n_itrs_per_episode: int = 1
    render_env: bool = False
    render_env_freq: int = 10


class DummyAgent(RLAgentBase):
    """The DummyGymAgent class. Dummy class to play
    with OpenAI-Gym environments

    """

    def __init__(self, algo_config: DummyAlgoConfig) -> None:
        super(DummyAgent, self).__init__(algo_config)

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """
        Do one step of the algorithm
        """

        episode_info = EpisodeInfo()
        done = False
        episode_reward = 0
        counter = 0

        for episode_itr in range(self.config.n_itrs_per_episode):

            action = env.sample_action()
            time_step = env.step(action=action)
            episode_reward += time_step.reward

            if self.config.render_env and episode_idx % self.config.render_env_freq == 0:
                env.render(mode="human")

            if done:
                break

            counter += 1

        episode_info.episode_reward = episode_reward
        episode_info.episode_iterations = counter
        return episode_info

    def play(self, env: Env, criterion: Criterion) -> PlayInfo:
        """Play the trained agent on the given environment

        Parameters
        ----------

        env: The environment to play on
        criterion: Specifies the criteria such that the play stops

        Returns
        -------

        An instance of PlayInfo

        """
        pass

    def on_state(self, state: State) -> Any:
        """Get an agent specific result e.g. an action or
        the state value

        Parameters
        ----------

        state: The state instance presented to the agent

        Returns
        -------

        An agent specific result e.g. an action or
        the state value

        """
        pass

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
        pass

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
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
        pass

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs after
        the iterations are finished

        Parameters
        ----------
        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        pass
