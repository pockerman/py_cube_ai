"""
Experience replay buffer
"""
import numpy as np
import random
from typing import Any, List
from collections import namedtuple, deque
import torch

from src.utils.exceptions import InvalidParameterValue

ExperienceTuple = namedtuple("ExperienceTuple", field_names=["state", "action",
                                                             "reward", "next_state", "done", "info"])


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

    TUPLE_NAMES = ["state", "action", "reward", "next_state", "done", "info"]

    def __init__(self, buffer_size: int) -> None:

        self.capacity: int = buffer_size
        self._memory = deque(maxlen=buffer_size)

    def add(self,  state: Any, action: Any, reward: float,
            next_state: Any, done: bool, info: dict = {}) -> None:
        """
        Add a new experience to memory.
        """
        e = ExperienceTuple(state, action, reward, next_state, done, info)
        self._memory.append(e)

    def sample(self, batch_size: int) -> list:
        """
        Randomly sample a batch of experiences from memory.
        """
        return random.sample(self._memory, k=batch_size)

    def __getitem__(self, name_attr: str) -> List:
        """
        Return the full batch of the name_attr attribute
        :param item:
        :return:
        """

        if name_attr not in ReplayBuffer.TUPLE_NAMES:
            raise InvalidParameterValue(param_name=name_attr, param_val=name_attr)

        batch = []
        for item in self._memory:

            if name_attr == "action":
                batch.append(item.action)
            elif name_attr == "state":
                batch.append(item.state)
            elif name_attr == "next_state":
                batch.append(item.next_state)
            elif name_attr == "reward":
                batch.append(item.reward)
            elif name_attr == "done":
                batch.append(item.done)
            elif name_attr == "info":
                batch.append(item.info)

        return batch

    def get_item_as_torch_tensor(self, name_attr) -> torch.Tensor:
        """
        Returns a torch.Tensor representation of the
        the named item
        :param name_attr:
        :return:
        """
        items = self[name_attr]

        # convert to np.array to avoid pytorch warning
        return torch.Tensor(np.array(items))

    def reinit(self) -> None:
        """
        Reinitialize the internal buffer
        :return:
        """
        self._memory = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        """
        Return the current size of internal memory.
        """
        return len(self._memory)



