import numpy as np
from typing import TypeVar

from src.policies.policy_adaptor_base import PolicyAdaptorBase

PolicyBase = TypeVar('PolicyBase')


class MaxActionEqualProbabilityAdaptorPolicy(PolicyAdaptorBase):
    """
    Given a policy a state
    and the state actions adapts the given policy for the
    given state by choosing the best action. Effectively the
    adaptor puts equal probability on maximizing actions
    """

    def __init__(self) -> None:
        super(MaxActionEqualProbabilityAdaptorPolicy, self).__init__()

    def __call__(self, policy: PolicyBase, *args, **kwargs) -> PolicyBase:

        s: int = kwargs["s"]
        state_actions: np.ndarray = kwargs["state_actions"]

        best_a = np.argwhere(state_actions == np.max(state_actions)).flatten()
        policy[s] = np.sum([np.eye(policy.env.action_space.n)[i] for i in best_a], axis=0) / len(best_a)
        return policy
