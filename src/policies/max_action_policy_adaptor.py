import numpy as np
from policies.policy_base import PolicyBase
from policies.policy_adaptor_base import PolicyAdaptorBase


class MaxActionPolicyAdaptor(PolicyAdaptorBase):
    """
    MaxActionPolicy class. Given the state
    """
    def __init__(self):
       super(MaxActionPolicyAdaptor, self).__init__()

    def __call__(self, state: int, state_actions: np.ndarray,
                 policy: np.ndarray) -> PolicyBase:
        policy[state][np.argmax(state_actions)] = 1
        return policy