"""
Experience replay buffere
"""

from typing import Any


class ReplayBuffer(object):
    """
    Fixed size replay buffer
    """
    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size


    def add(self,  state: Any, action: Any, reward: float,
            next_state: Any, done: bool):
        pass

    def sample(self) -> Any:
        pass

    def __len__(self):
        pass



