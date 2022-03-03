"""The module agent_base. Specifies the interface that
an agent should implement

"""

import abc
from typing import TypeVar, Any


Policy = TypeVar('Policy')
State = TypeVar('State')
Env = TypeVar('Env')
Criteria = TypeVar('Criteria')
PlayInfo = TypeVar('PlayInfo')


class AgentBase(metaclass=abc.ABCMeta):
    """The AgentBase class.

    """

    def __init__(self, policy: Policy):
        self.policy: Policy = policy

    def play(self, env: Env, criteria: Criteria) -> PlayInfo:
        """Apply the agent on the environment until the specified
        criteria is fulfilled

        Parameters
        ----------

        env: The environment
        criteria: The criteria

        Returns
        -------

        None
        """

    def on_state(self, state: State) -> Any:
        """

        Parameters
        ----------
        state

        Returns
        -------

        """