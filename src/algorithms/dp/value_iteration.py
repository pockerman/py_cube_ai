"""The value_iteration module. Provides a simple
implementation of value iteration algorithm

"""

import numpy as np
from typing import TypeVar

from src.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from src.algorithms.dp.policy_improvement import PolicyImprovement
from src.algorithms.dp.utils import state_actions_from_v as q_s_a
from src.policies.policy_adaptor_base import PolicyAdaptorBase
from src.utils.mixins import WithValueTableMixin
from src.utils.episode_info import EpisodeInfo
from src.worlds.world_helpers import n_states

Env = TypeVar("Env")


class ValueIteration(DPAlgoBase, WithValueTableMixin):
    """The class ValueIteration implements the value iteration
    algorithm
    """

    def __init__(self, algo_config: DPAlgoConfig,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        """
        Constructor

        Parameters
        ----------

        algo_config Algorithm configuration
        policy_adaptor How the underlying policy is adapted

        """
        super(ValueIteration, self).__init__(algo_config=algo_config)
        self._p_imprv = PolicyImprovement(algo_in=algo_config,
                                          v=None,  policy_adaptor=policy_adaptor)

    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """

        # zero the value function
        self.v = np.zeros(n_states(env))
        self._p_imprv.v = self.v

    def on_state(self, state: int) -> float:
        """
        Returns the value of the value function
        on the given state

        Parameters
        ----------
        state The state given

        Returns
        -------

        The value of the value function

        """

        return self.v[state]

    def on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """
        Train the agent on the environment at the given episode.

        Parameters
        ----------

        env The environment to train on
        episode_idx The episode index
        options Any options passes by the client code

        Returns
        -------

        An instance of the EpisodeInfo class

        """

        delta = 0

        n_states_ = n_states(env)
        for s in range(n_states_):
            v = self.v[s]
            self.v[s] = max(q_s_a(env=env, v=self.v, state=s, gamma=self.gamma))
            delta = max(delta, abs(self.v[s] - v))

        episode_info = EpisodeInfo()
        if delta < self.config.tolerance:
            episode_info.info["break_training"] = True

        self._p_imprv.v = self.v
        self._p_imprv.on_training_episode(env, episode_idx, **options)
        self.policy = self._p_imprv.policy
        return episode_info




