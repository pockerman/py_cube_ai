"""Module policy_improvement. Implements the policy improvement
algorithm as this is described in the book


http://incompleteideas.net/book/RLbook2020.pdf

"""

import numpy as np
from typing import TypeVar
from pycubeai.algorithms.dp.dp_algorithm_base import DPAlgoBase, DPAlgoConfig
from pycubeai.algorithms.dp.utils import state_actions_from_v as q_s_a

from pycubeai.utils.mixins import WithValueTableMixin
from pycubeai.utils.episode_info import EpisodeInfo
from pycubeai.worlds.world_helpers import n_states
from pycubeai.utils.wrappers import time_func_wrapper

PolicyAdaptor = TypeVar('PolicyAdaptor')
Env = TypeVar('Env')


class PolicyImprovement(DPAlgoBase, WithValueTableMixin):
    """Implementation of policy improvement
    """

    def __init__(self, algo_config: DPAlgoConfig, v: np.array,  policy_adaptor: PolicyAdaptor) -> None:
        """Constructor. Initialize an algorithm instance using the
        configuration instance the value-function and the object that
        adapts the policy

        Parameters
        ----------

        algo_config: Algorithm configuration
        v: The value function to use
        policy_adaptor: The object responsible to adapt the policy

        """

        super(PolicyImprovement, self).__init__(algo_config=algo_config)
        self.v = v
        self.policy_adaptor: PolicyAdaptor = policy_adaptor

    @time_func_wrapper(show_time=False)
    def do_on_training_episode(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
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
        episode_reward = 0.0
        episode_itrs = 0

        n_states_ = n_states(env)

        for s in range(n_states_):
            state_actions = q_s_a(env=env, v=self.v, state=s, gamma=self.gamma)
            self.policy = self.policy_adaptor.adapt(self.policy, state_actions=state_actions,  s=s)

        episode_info = EpisodeInfo(episode_index=episode_idx, episode_iterations=episode_itrs,
                                   episode_reward=episode_reward)
        episode_info.info["break_training"] = True
        return episode_info
