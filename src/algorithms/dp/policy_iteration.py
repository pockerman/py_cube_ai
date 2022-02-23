"""Implementation of Policy iteration algorithm. In policy
iteration at each step we do one policy evaluation and one policy
improvement.

"""

import numpy as np
from typing import TypeVar
import copy


from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from src.algorithms.dp.iterative_policy_evaluation import IterativePolicyEvaluator
from src.algorithms.dp.policy_improvement import PolicyImprovement
from src.policies.policy_adaptor_base import PolicyAdaptorBase
from src.utils.episode_info import EpisodeInfo
from src.utils.wrappers import time_fn


Env = TypeVar('Env')
Policy = TypeVar('Policy')


class PolicyIteration(DPAlgoBase):
    """Policy iteration class
    """

    def __init__(self, algo_config: DPAlgoConfig,  policy_adaptor: PolicyAdaptorBase):
        """
        Constructor.

        Parameters
        ----------

        algo_config Configuration for the algorithm
        policy_adaptor How the policy should be adapted

        """
        super(PolicyIteration, self).__init__(algo_config=algo_config)

        self._p_eval = IterativePolicyEvaluator(algo_config=algo_config)
        self._p_imprv = PolicyImprovement(algo_in=algo_config, v=self._p_eval.v, policy_adaptor=policy_adaptor)

    @property
    def v(self) -> np.array:
        return self._p_imprv.v

    @property
    def policy(self) -> Policy:
        """
        Get the trained policy

        Returns
        -------

        An instance of Policy

        """
        return self._p_eval.policy

    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """Execute any actions the algorithm needs before
        starting the iterations

        Parameters
        ----------

        env: The environment to train on
        options: Any options passed by the application

        Returns
        -------

        None

        """

        # call the base class version
        super(PolicyIteration, self).actions_before_training_begins(env,  **options)
        self._p_eval.actions_before_training_begins(env,  **options)
        self._p_imprv.actions_before_training_begins(env,  **options)

    def actions_after_training_ends(self, env: Env, **options) -> None:
        """Any actions the algorithm should perform after the training ends

        Parameters
        ----------

        env The environment the agent is trained on
        options Any options passed by the client code

        Returns
        -------

        None

        """
        super(PolicyIteration, self).actions_after_training_ends(env,  **options)
        self._p_eval.actions_after_training_ends(env,  **options)
        self._p_imprv.actions_after_training_ends(env,  **options)

    @time_fn
    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """

        Train the agent on the given environment and the given episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The episode index
        options: Various data passed by the client

        Returns
        -------

        Instance of EpisodeInfo class

        """

        # make a copy of the policy already obtained
        old_policy = copy.deepcopy(self._p_eval.policy)

        # evaluate the policy
        train_info = self._p_eval.on_training_episode(env, episode_idx, **options)

        # update the value function to
        # improve for
        self._p_imprv.v = self._p_eval.v

        # improve the policy
        self._p_imprv.on_training_episode(env, episode_idx, **options)

        new_policy = self._p_imprv.policy

        episode_info = EpisodeInfo()
        episode_info.episode_reward = train_info.episode_reward
        episode_info.episode_iterations = train_info.episode_iterations

        # check of the two policies are the same
        if old_policy == new_policy:
            episode_info.info["break_training"] = True

        self._p_eval.policy = new_policy
        return episode_info
