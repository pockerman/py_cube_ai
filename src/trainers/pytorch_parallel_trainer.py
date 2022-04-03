"""
PyTorch trainer
"""
import copy
from typing import TypeVar
from src.utils.train_info import TrainInfo
from src.parallel_utils.torch_processes_handler import TorchProcsHandler
from src.trainers.rl_agent_trainer_base import RLAgentTrainerBase, RLAgentTrainerConfig
from src.utils.exceptions import InvalidParameterValue
from src.optimization.pytorch_optimizer_builder import pytorch_optimizer_builder


Env = TypeVar('Env')
Model = TypeVar('Model')
PlayInfo = TypeVar('PlayInfo')
State = TypeVar('State')
Action = TypeVar('Action')


class PyTorchParallelTrainerConfig(RLAgentTrainerConfig):
    def __init__(self):
        self.n_procs: int = 0
        self.model: Model = None
        self.learning_rate: float = 0.01
        self.gamma: float = 0.99


class PyTorchParallelTrainer(RLAgentTrainerBase):

    @staticmethod
    def worker(env: Env, config: PyTorchParallelTrainerConfig):

        # copy the environment for the process
        env = copy.deepcopy(env)
        env.reset()

        # create the optimizer to use
        worker_opt = pytorch_optimizer_builder(config.model.config.opt_type,
                                               model_params=config.model.parameters(),
                                               **{"learning_rate": config.learning_rate})

        for episode in range(config.n_episodes):
            worker_opt.zero_grad()
            episode_info = config.model.on_training_episode(env, episode_idx=episode)

            # values, logprobs, rewards = run_episode(worker_env, model)  # B
            # actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)  # C
            # counter.value = counter.value + 1  # D

    def __init__(self, config: PyTorchParallelTrainerConfig) -> None:
        super(PyTorchParallelTrainer, self).__init__()
        self.config = config
        self.procs_handler: TorchProcsHandler = None

    def get_configuration(self) -> PyTorchParallelTrainerConfig:
        """
        Returns the configuration of the agent
        :return:
        """
        return self.config

    def train(self, env: Env, **options) -> TrainInfo:
        """

        :return:
        :rtype:
        """
        train_info = TrainInfo()
        self.actions_before_training_begins(env)
        self.procs_handler.create_and_start(self.worker, *(env, self.config))
        self.procs_handler.join_and_terminate()
        self.actions_after_training_ends(env)
        return train_info

    def actions_before_training_begins(self, env: Env, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param env:
        :param episode_idx:
        :param info:
        :return:
        """
        self._validate(env)
        self.procs_handler: TorchProcsHandler = TorchProcsHandler(n_procs=self.config.n_procs)
        self.config.model.actions_before_training_begins(env, episode_idx=0, **{"gamma": self.config.gamma,
                                                                                "learning_rate": self.config.learning_rate})

    def actions_before_episode_begins(self, env: Env, **info) -> None:
        """
        Execute any actions the algorithm needs before
        starting the episode
        :param options:
        :return:
        """
        pass

    def actions_after_episode_ends(self, env:Env, **info) -> None:
        """
        Execute any actions the algorithm needs after
        ending the episode
        :param options:
        :return:
        """
        pass

    def actions_after_training_ends(self, env: Env, **info) -> None:
        """
        Execute any actions the algorithm needs after
        the iterations are finished
        """
        pass

    def _validate(self, env: Env) -> None:

        if self.config.n_episodes <=0:
            raise InvalidParameterValue(param_name="n_episodes", param_val=self.config.n_episodes)

        if env is None:
            raise InvalidParameterValue(param_name="env", param_val="None")

        if self.config.model is None:
            raise InvalidParameterValue(param_name="model", param_val="None")