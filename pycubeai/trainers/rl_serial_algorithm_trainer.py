"""Module rl_serial_algorithm_trainer
Specifies an interface to train serial RL algorithms

"""
from typing import TypeVar
from dataclasses import dataclass

from pycubeai.trainers.rl_agent_trainer_base import RLAgentTrainerBase
from pycubeai.utils.iteration_controller import ItrControlResult, IterationController
from pycubeai.utils.wrappers import time_fn
from pycubeai.utils import INFO

Env = TypeVar('Env')
Algorithm = TypeVar('Algorithm')


@dataclass(init=True, repr=True, )
class RLSerialTrainerConfig(object):
    """The RLSerialTrainerConfig class.
    Configuration class for RLSerialAgentTrainer

    """

    n_episodes: int = 0
    tolerance: float = 1.0e-8
    output_msg_frequency: int = -1
    render_env: bool = False
    render_env_freq = 100


class RLSerialAgentTrainer(RLAgentTrainerBase):
    """The RLSerialAgentTrainer class handles the training
    for serial reinforcement learning agents

    """

    def __init__(self, config: RLSerialTrainerConfig, algorithm: Algorithm):
        """Constructor

        Parameters
        ----------
        config: Configuration for the trainer
        algorithm: The algorithm to train

        """
        super(RLSerialAgentTrainer, self).__init__(config=config, algorithm=algorithm)
        self._itr_ctrl = IterationController(tol=config.tolerance, n_max_itrs=config.n_episodes)
        self.break_training_flag: bool = False
        self.rewards = []
        self.iterations_per_episode = []

    @property
    def current_episode_index(self) -> int:
        return self._itr_ctrl.current_itr_counter - 1

    @property
    def itr_control(self) -> IterationController:
        return self._itr_ctrl

    @property
    def avg_rewards(self) -> list:
        avgs = [reward / itr for reward, itr in zip(self.rewards, self.iterations_per_episode)]
        return avgs

    @time_fn
    def train(self, env: Env, **options) -> ItrControlResult:
        """Train the algorithm on the given environment

        Parameters
        ----------
        env: The environment to train the algorithm
        options: Any options passed by the client

        Returns
        -------

        An instance of ItrControlResult class
        """

        itr_ctrl_rsult = ItrControlResult(tol=self._itr_ctrl.tolerance,
                                          residual=self._itr_ctrl.residual,
                                          n_itrs=0, n_max_itrs=self._itr_ctrl.n_max_itrs,
                                          n_procs=1)

        self.actions_before_training_begins(env, **options)

        counter = 0
        while self._itr_ctrl.continue_itrs():

            remains = counter % self.trainer_config.output_msg_frequency
            if self.trainer_config.output_msg_frequency != -1 and remains == 0:
                print("{0}: Episode {1} of {2}, ({3}% done)".format(INFO, self.current_episode_index,
                                                                    self.itr_control.n_max_itrs,
                                                                    (
                                                                                self.itr_control.current_itr_counter / self.itr_control.n_max_itrs) * 100.0))
            self.actions_before_episode_begins(env, **options)
            episode_info = self.algorithm.on_training_episode(env, self.current_episode_index, **options)

            if self.trainer_config.output_msg_frequency != -1 and remains == 0:
                print("{0} {1}".format(INFO, episode_info))

            self.rewards.append(episode_info.episode_reward)
            self.iterations_per_episode.append(episode_info.episode_iterations)

            if "break_training" in episode_info.info and \
                    episode_info.info["break_training"] is True:
                self.break_training_flag = True

            self.actions_after_episode_ends(env, **options)
            counter += 1

            # check if the break training flag
            # has been set and break
            if self.break_training_flag:
                print("{0}: On Episode {1} the break training "
                      "flag was set. Stop training".format(INFO, self.current_episode_index))

                # if we get here then assume we have converged
                self._itr_ctrl.residual = self.trainer_config.tolerance * 1.0e-2
                break

        self.actions_after_training_ends(env, **options)

        # update the control result
        itr_ctrl_rsult.n_itrs = self._itr_ctrl.current_itr_counter
        itr_ctrl_rsult.residual = self._itr_ctrl.residual

        return itr_ctrl_rsult

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs before training starts

        Parameters
        ----------
        env: The environment to train the algorithm
        options: Any options passed by the client

        Returns
        -------

        None
        """

        env.reset()
        self.rewards = []
        self.iterations_per_episode = []
        self.algorithm.actions_before_training_begins(env, **options)

    def actions_before_episode_begins(self, env: Env, **options) -> None:
        """Execute any needed actions  before the
         training episode begins

        Parameters
        ----------
        env: The environment to train the algorithm
        options: Any options passed by the client

        Returns
        -------
        None

        """
        self.algorithm.actions_before_episode_begins(env, self.current_episode_index, **options)

    def actions_after_episode_ends(self, env: Env, **options) -> None:
        """Execute any needed actions  after the
                 training episode begins

        Parameters
        ----------

        env: The environment to train the algorithm
        options: Any options passed by the client

        Returns
        -------
        None

        """
        self.algorithm.actions_after_episode_ends(env, self.current_episode_index, **options)

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """Execute any actions the algorithm needs after
        training episodes are finished

        Parameters
        ----------
        env: The environment to train the algorithm
        options: Any options passed by the client

        Returns
        -------

        None
        """

        self.algorithm.actions_after_training_ends(env, **options)
