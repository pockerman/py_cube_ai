from typing import TypeVar
from src.algorithms.rl_agent_trainer_base import RLAgentTrainerBase
from src.utils.iteration_controller import ItrControlResult, IterationController
from src.utils.wrappers import time_fn
from src.utils import INFO

Env = TypeVar('Env')
Agent = TypeVar('Agent')


class RLSerialTrainerConfig(object):
    """
    The AlgoInput class. Wraps the common input
    that most algorithms use. Concrete algorithms
    can extend this class to accommodate their specific
    input as well
    """

    def __init__(self):
        self.n_episodes: int = 0
        self.tolerance: float = 1.0e-8

        # negative means do not output
        # any messages
        self.output_msg_frequency: int = -1
        self.render_env: bool = False
        self.render_env_freq = 100


class RLSerialAgentTrainer(RLAgentTrainerBase):
    """
    The RLSerialAgentTrainer class handles the training
    for serial reinforcement learning agents
    """

    def __init__(self, config: RLSerialTrainerConfig, agent: Agent) -> None:
        super(RLSerialAgentTrainer, self).__init__(config=config, agent=agent)
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

    @time_fn
    def train(self, env: Env, **options) -> ItrControlResult:
        """
        :return:
        :rtype:
        """
        self.actions_before_training_begins(env, **options)

        itr_ctrl_rsult = ItrControlResult(tol=self._itr_ctrl.tolerance,
                                          residual=self._itr_ctrl.residual,
                                          n_itrs=0, n_max_itrs=self._itr_ctrl.n_max_itrs,
                                          n_procs=1)

        self.actions_before_training_begins(env, **options)

        counter = 0
        while self._itr_ctrl.continue_itrs():

            if self.trainer_config.output_msg_frequency != -1:
                remains = counter % self.trainer_config.output_msg_frequency
                if remains == 0:
                    print("{0}: Episode {1} of {2}, ({3}% done)".format(INFO, self.current_episode_index,
                                                                        self.itr_control.n_max_itrs,
                                                                        (self.itr_control.current_itr_counter / self.itr_control.n_max_itrs) * 100.0))
            self.actions_before_episode_begins(env, **options)
            episode_info = self.agent.on_training_episode(env, **{"episode_idx": self.current_episode_index})
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
                break

        self.actions_after_training_ends(env, **options)

        # update the control result
        itr_ctrl_rsult.n_itrs = self._itr_ctrl.current_itr_counter
        itr_ctrl_rsult.residual = self._itr_ctrl.residual

        return itr_ctrl_rsult

    def actions_before_training_begins(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
        """
        env.reset()
        self.rewards = []
        self.iterations_per_episode = []
        self.agent.actions_before_training_begins(env, self.current_episode_index, **options)

    def actions_before_episode_begins(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        self.agent.actions_before_episode_begins(env, self.current_episode_index, **options)

    def actions_after_episode_ends(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode
        :param options:
        :return:
        """
        self.agent.actions_after_episode_ends(env, self.current_episode_index, **options)

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        self.agent.actions_after_training_ends(env, self.current_episode_index, **options)
