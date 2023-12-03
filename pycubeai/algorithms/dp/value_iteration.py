"""The value_iteration module. Provides a simple
implementation of value iteration algorithm

"""

import numpy as np
from typing import TypeVar

from pycubeai.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from pycubeai.algorithms.dp.policy_improvement import PolicyImprovement
from pycubeai.algorithms.dp.utils import state_actions_from_v as q_s_a
from pycubeai.policies.policy_adaptor_base import PolicyAdaptorBase
from pycubeai.utils.mixins import WithValueTableMixin
from pycubeai.utils.episode_info import EpisodeInfo
from pycubeai.worlds.world_helpers import n_states
from pycubeai.utils.wrappers import time_func_wrapper

Env = TypeVar("Env")


class ValueIteration(DPAlgoBase, WithValueTableMixin):
    """The class ValueIteration implements the value iteration
    algorithm
    """

    def __init__(self, algo_config: DPAlgoConfig,
                 policy_adaptor: PolicyAdaptorBase) -> None:
        """Constructor

        Parameters
        ----------

        algo_config Algorithm configuration
        policy_adaptor How the underlying policy is adapted

        """
        super(ValueIteration, self).__init__(algo_config=algo_config)
        self._p_imprv = PolicyImprovement(algo_config=algo_config,
                                          v=None,  policy_adaptor=policy_adaptor)

    def actions_before_training_begins(self, env: Env,  **options) -> None:
        """Execute any actions the algorithm needs before
        starting the iterations

        Parameters
        ----------
        env: The environment to train on
        options: Any options passed by the client code

        Returns
        -------

        None

        """

        super(ValueIteration, self).actions_before_training_begins(env, **options)
        # zero the value function
        self.v = np.zeros(n_states(env))
        self._p_imprv.v = self.v

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

        delta = 0.0

        n_states_ = n_states(env)
        for s in range(n_states_):
            v = self.v[s]
            self.v[s] = max(q_s_a(env=env, v=self.v, state=s, gamma=self.gamma))
            delta = max(delta, abs(self.v[s] - v))

        episode_info = EpisodeInfo(episode_index=episode_idx)
        if delta < self.config.tolerance:
            episode_info.info["break_training"] = True

        self._p_imprv.v = self.v
        self._p_imprv.on_training_episode(env, episode_idx, **options)
        self.policy = self._p_imprv.policy
        return episode_info




