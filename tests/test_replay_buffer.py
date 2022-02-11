"""
Unit-tests for ReplayBuffer
"""
import unittest
import torch
from src.utils.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):


    def test_constructor(self):
        buffer = ReplayBuffer(buffer_size=100)
        self.assertEqual(100, buffer.capacity)
        self.assertEqual(0, len(buffer))

    def test_add(self):
        buffer = ReplayBuffer(buffer_size=100)
        buffer.add(state=1, action=0, reward=1.0,
                   next_state=2, done=False,info={})
        self.assertEqual(1, len(buffer))

    def test_getitem(self):
        buffer = ReplayBuffer(buffer_size=100)
        buffer.add(state=1, action=0, reward=1.0,
                   next_state=2, done=False, info={})

        actions = buffer["action"]
        self.assertEqual(1, len(actions))
        self.assertEqual(0, actions[0])

    def test_reinit(self):
        buffer = ReplayBuffer(buffer_size=100)
        buffer.add(state=1, action=0, reward=1.0,
                   next_state=2, done=False, info={})
        self.assertEqual(1, len(buffer))

        buffer.reinit()
        self.assertEqual(0, len(buffer))

    def test_item_as_torch_tensor(self):
        buffer = ReplayBuffer(buffer_size=100)
        buffer.add(state=1, action=0, reward=1.0,
                   next_state=2, done=False, info={})

        tensor = buffer.get_item_as_torch_tensor("state")

        self.assertEqual(1, tensor.shape[0])


if __name__ == '__main__':
    unittest.main()

