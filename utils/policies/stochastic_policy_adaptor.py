import numpy as np

from utils.policies.policy_base import PolicyBase
from utils.policies.policy_adaptor_base import PolicyAdaptorBase


class StochasticAdaptorPolicy(PolicyAdaptorBase):
    """
    Stochastic policy adaptor. Given a policy a state
    and the state actions adapts the given policy for the
    given state
    """

    def __init__(self) -> None:
        super(StochasticAdaptorPolicy, self).__init__()

    def __call__(self, s: int, state_actions: np.ndarray, policy: PolicyBase) -> PolicyBase:
        best_a = np.argwhere(state_actions == np.max(state_actions)).flatten()
        policy[s] = np.sum([np.eye(policy.env.action_space.n)[i] for i in best_a], axis=0) / len(best_a)
        return policy