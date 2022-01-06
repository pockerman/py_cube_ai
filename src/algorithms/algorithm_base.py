"""
Helper class for iterative algorithms
"""

from enum import Enum
import abc
from abc import abstractmethod, ABC
from typing import Any, TypeVar

from src.utils.exceptions import InvalidParameterValue
from src.utils.wrappers import time_fn
from src.utils.iteration_controller import ItrControlResult, IterationController
from src.utils import INFO


Env = TypeVar("Env")
AlgoInput = TypeVar("AlgoInput")


class AlgorithmBase(ABC):
    """
    Base class for deriving algorithms
    """

    def __init__(self, algo_in: AlgoInput) -> None:
        super(AlgorithmBase, self).__init__()
        self._itr_ctrl = IterationController(tol=algo_in.tolerance, n_max_itrs=algo_in.n_episodes)
        self._train_env = algo_in.train_env
        self._state = None
        self.render_env = algo_in.render_env
        self.render_env_freq = algo_in.render_env_freq
        self.output_msg_frequency: int = algo_in.output_freq

    def __call__(self, **options) -> ItrControlResult:
        """
        Make the module callable
        """
        return self.train(**options)

    @property
    def state(self) -> Any:
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def train_env(self) -> Any:
        """
        Returns the environment used for training
        """
        return self._train_env

    @train_env.setter
    def train_env(self, value: Any) -> None:
        self._train_env = value

    @property
    def itr_control(self) -> IterationController:
        return self._itr_ctrl

    @property
    def n_episodes(self) -> int:
        return self.itr_control.n_max_itrs

    @property
    def current_episode_index(self) -> int:
        return self._itr_ctrl.current_itr_counter - 1

    def reset(self) -> None:
        """
        Reset the underlying data
        """
        self._state = self.train_env.reset()
        self._itr_ctrl.reset()

    @time_fn
    def train(self, **options) -> ItrControlResult:

        """
        Iterate to train the agent
        """

        itr_ctrl_rsult = ItrControlResult(tol=self._itr_ctrl.tolerance,
                                          residual=self._itr_ctrl.residual,
                                          n_itrs=0, n_max_itrs=self._itr_ctrl.n_max_itrs,
                                          n_procs=1)

        self.actions_before_training_begins(**options)

        counter = 0
        while self._itr_ctrl.continue_itrs():

            remains = counter % self.output_msg_frequency
            if remains == 0:

                print("{0}: Episode {1} of {2}, ({3}% done)".format(INFO, self.current_episode_index,
                                                                    self.itr_control.n_max_itrs,
                                                                    (self._itr_ctrl.current_itr_counter / self.itr_control.n_max_itrs)*100.0))
            self.actions_before_episode_begins(**options)
            self.on_episode(**options)
            self.actions_after_episode_ends(**options)
            counter += 1

        self.actions_after_training_ends(**options)

        # update the control result
        itr_ctrl_rsult.n_itrs = self._itr_ctrl.current_itr_counter
        itr_ctrl_rsult.residual = self._itr_ctrl.residual

        return itr_ctrl_rsult

    def actions_before_episode_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return: None
        """
        self._state = self.train_env.reset()

    def actions_after_episode_ends(self, **options):
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        pass

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the training
        """

        if self.train_env is None:
            raise ValueError("Environment is None")

        self.reset()

        if self.n_episodes == 0:
            raise InvalidParameterValue(param_name="n_episodes", param_val=self.n_episodes)

    @abstractmethod
    def actions_after_training_ends(self, **options) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        raise NotImplementedError("The function must be overridden")

    @abstractmethod
    def on_episode(self, **options) -> None:
        """
        Do one step of the algorithm
        """
        raise NotImplementedError("The function must be overridden")





