from typing import TypeVar, Any
import abc

Env = TypeVar('Env')
State = TypeVar('State')
Action = TypeVar('Action')
TiledState = TypeVar('TiledState')


class TiledEnvWrapper(metaclass=abc.ABCMeta):
    """
    Abstract wrapper class to the given environment for tiled based
    algorithms
    """

    EPISODIC_CONSTRAINT = True
    HAS_DISCRETE_STATES = True
    HAS_DISCRETE_ACTIONS = True

    # consraints list
    CONSTRAINTS = [EPISODIC_CONSTRAINT, ]

    def __init__(self, env: Env, n_actions: int, n_states: int) -> None:
        self.raw_env = env
        self.n_actions = n_actions
        self.n_states = n_states
        self.discrete_observation_space = []

    @property
    def action_space(self) -> Any:
        return self.raw_env.action_space

    def reset(self) -> State:
        """
        Reset the underlying environment
        :return: State
        """
        return self.raw_env.reset()

    def get_property(self, prop_str: str) -> Any:
        """
        Get the property in the underlying environment
        :param prop_str:
        :return:
        """
        return self.raw_env.__dict__[prop_str]

    def step(self, action: Action):
        """
        Step in the environment
        :param action:
        :return:
        """
        return self.raw_env.step(action=action)

    def close(self) -> None:
        """
        Close the environment
        :return:
        """
        self.raw_env.close()

    def render(self, mode='human'):
        """
        Render the environment
        :param mode:
        :return:
        """
        return self.raw_env.render(mode=mode)

    def get_state_from_obs(self, obs: State) -> TiledState:
        """
        Returns the state idx given the raw observation
        :param obs: Raw observation
        :return: tiled state
        """
        return self.get_tiled_state(obs=obs, action=None)

    @abc.abstractmethod
    def get_tiled_state(self, obs: State, action: Action) -> TiledState:
        """
        Returns the tiled states for the given observation
        :param obs: The observation to be tiled
        :param action: The action corresponding to the states
        :return: TiledState
        """

    @abc.abstractmethod
    def create_bins(self) -> None:
        """
        Create the bins that the state variables of
        the underlying environment will be distributed
        :return: A list of bins for every state variable
        """

    @abc.abstractmethod
    def create_state_space(self) -> None:
        """
        Creates the discrete state space.
        :return:
        """

