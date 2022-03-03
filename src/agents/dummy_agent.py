"""The module dummy_agent. Specifies a dummy agent

"""

import random
from typing import TypeVar, Any
from src.utils.play_info import PlayInfo
from src.agents.agent_base import AgentBase

Env = TypeVar('Env')
Criteria = TypeVar('Criteria')
State = TypeVar('State')


class DummyAgent(AgentBase):
    def __init__(self, policy):
        super(DummyAgent, self).__init__(policy=policy)

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


        action_idx = 0

        while criteria.continue_itrs():
            action = self.policy[action_idx]
            time_step = env.step(action)
            env.render(mode="human")

            if time_step.done:
                env.reset()
                action_idx = 0
            else:
                action_idx += 1

        return PlayInfo()

    def on_state(self, state: State) -> Any:
        """Randomly select an action to retrun

        Parameters
        ----------
        state: The state the agent observes

        Returns
        -------

        """
        idx = random.randint(0, len(self.policy))
        return self.policy[idx]
