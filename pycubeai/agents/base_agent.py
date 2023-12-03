"""
Class for deriving agents
"""

from abc import abstractmethod, ABC


class BaseAgent(ABC):

    def __init__(self, device: str) -> None:
        super(BaseAgent, self).__init__()
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    @abstractmethod
    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError
