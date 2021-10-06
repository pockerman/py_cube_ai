"""
Experience replay buffere
"""
import numpy as np
import random
from typing import Any
from collections import namedtuple, deque
import torch


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class ReplayBuffer(object):
    """
    Fixed size replay buffer
    """
    def __init__(self, batch_size: int, action_size: int,
                 buffer_size: int, seed: int, device: str='cpu') -> None:
        self._batch_size = batch_size
        self._action_size = action_size
        self._memory = deque(maxlen=buffer_size)
        self._seed = seed
        self._device = device

        random.seed(self._seed)

    def add(self,  state: Any, action: Any,
            reward: float, next_state: Any, done: bool):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self._memory.append(e)

    def sample(self) -> list:
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self._memory, k=self._batch_size)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self._memory)



