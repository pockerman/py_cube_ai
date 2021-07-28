"""
Experience replay buffere
"""
import numpy as np
import random
from typing import Any
from collections import namedtuple, deque
import torch


class ReplayBuffer(object):
    """
    Fixed size replay buffer
    """
    def __init__(self, batch_size: int, action_size: int,
                 buffer_size: int, seed: int, device: str='cpu') -> None:
        self._batch_size = batch_size
        self._action_size = action_size
        self._memory = deque(maxlen=buffer_size)
        self._experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._seed = seed
        self._device = device

        random.seed(self._seed)

    def add(self,  state: Any, action: Any,
            reward: float, next_state: Any, done: bool):
        """Add a new experience to memory."""
        e = self._experience(state, action, reward, next_state, done)
        self._memory.append(e)

    def sample(self) -> tuple:
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self._memory, k=self._batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self._device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self._device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self._memory)



