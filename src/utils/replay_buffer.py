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
    Fixed size replay buffer.
    The buffer is represented by using a deque from Python’s built-in collections library.
    This is basically a list that we can set a maximum size.  If we try to add a new element whilst the list
    is already full, it will remove the first item in the list and add the new item to the end of the list.
    Hence new experiences  replace the oldest experiences.
    The experiences themselves are tuples of (state1, reward, action, state2, done) that we append to the replay deque
    and they are represented via the named tuple ExperienceTuple
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



