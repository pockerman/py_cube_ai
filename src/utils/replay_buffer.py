"""
Experience replay buffer
"""
import numpy as np
import random
from typing import Any
from collections import namedtuple, deque
import torch


ExperienceTuple = namedtuple("ExperienceTuple", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer(object):
    """
    Fixed size replay buffer
    """
    def __init__(self, buffer_size: int) -> None:

        self._memory = deque(maxlen=buffer_size)

    def add(self,  state: Any, action: Any, reward: float,
            next_state: Any, done: bool) -> None:
        """
        Add a new experience to memory.
        """
        e = ExperienceTuple(state, action, reward, next_state, done)
        self._memory.append(e)

    def sample(self, batch_size: int) -> list:
        """
        Randomly sample a batch of experiences from memory.
        """
        return random.sample(self._memory, k=batch_size)

    def __len__(self) -> int:
        """
        Return the current size of internal memory.
        """
        return len(self._memory)



