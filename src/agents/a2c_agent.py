"""
Agent class for A2C type agent
"""

import numpy as np
from agents.base_agent import BaseAgent
from networks.nn_base import NNBase
from agents.state_preprocessors import float32_preprocessor


class A2CAgent(BaseAgent):

    def __init__(self, net: NNBase, device: str) -> None:
        super(A2CAgent, self).__init__(device=device)
        self._net = net

    @property
    def net(self):
        return self._net

    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        logstd = self.net.logstd.data.cpu().numpy()
        rnd = np.random.normal(size=logstd.shape)
        actions = mu + np.exp(logstd) * rnd
        actions = np.clip(actions, -1, 1)
        return actions, agent_states