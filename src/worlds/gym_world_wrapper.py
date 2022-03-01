"""The module world_wrapper. Specifies a wrapper
for environments that don't return TimeStep when calling reset
or step

"""

from typing import TypeVar
from src.utils.time_step import TimeStep, StepType

Env = TypeVar('Env')
Action = TypeVar('Action')


class GymWorldWrapper(object):
    """The GymWorldWrapper wrapper for
    OpenAI-Gym environments

    """

    def __init__(self, gym_env: Env, discount: float = 0.1) -> None:
        """Constructor

        Parameters
        ----------
        gym_env: The raw OpenAI-Gym environment
        discount: The discount factor

        """
        self.gym_env = gym_env
        self.discount = discount

    def reset(self) -> TimeStep:
        """Reset the environment

        Returns
        -------

        An instance of the TimeStep class
        """
        observation = self.gym_env.reset()
        return TimeStep(step_type=StepType.FIRST, reward=0.0,
                        observation=observation, discount=self.discount, info={})

    def step(self, action: Action) -> TimeStep:
        """Step in the environment

        Parameters
        ----------
        action: The action to execute in the environment

        Returns
        -------

        An instance of the TimeStep class
        """
        observation, reward, done, info = self.gym_env.step(action)

        step_type = StepType.MID
        if done:
            step_type = StepType.LAST

        return TimeStep(step_type=step_type, reward=0.0,
                        observation=observation, discount=self.discount, info={})

    def sample_action(self) -> Action:
        """Sample an action from the observation space

        Returns
        -------

        The sampled action

        """
        return self.gym_env.action_space.sample()

    def render(self, mode: str) -> None:
        """Render the underlying environment

        Parameters
        ----------
        mode: Rendering mode. This depends on the
        underlying OpenAI-Gym environment

        Returns
        -------

        None
        """
        self.gym_env.render(mode)


