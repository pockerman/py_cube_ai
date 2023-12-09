"""
Helpers for DP algorithms
"""

import numpy as np
from typing import Any


def state_actions_from_v(env: Any, v: np.ndarray,
                         gamma: float, state: int) -> np.ndarray:
    """
    Given the state index returns the list of actions under the
    provided value functions
    """
    q = np.zeros(env.action_space.n)

    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[state][a]:
            q[a] += prob * (reward + gamma * v[next_state])
    return q


def q_from_v(env: Any, v: np.ndarray, gamma: float,) -> dict:
    """
    Returns the state-action value function for the
    approximated value function
    """
    q_map = {}

    for s in range(env.observation_space.n):
        q_map[s] = state_actions_from_v(env=env, v=v, gamma=gamma, state=s)
    return q_map