"""
Implementation of vanilla DQN.
Implementation refactored from
https://github.com/udacity/deep-reinforcement-learning
"""

from typing import Any
from collections import deque
import numpy as np
import random
import torch
import torch.nn.functional as F

from src.networks.nn_base import NNBase
from src.networks.utils import soft_network_update
from src.utils.replay_buffer import ReplayBuffer
from src.algorithms.algorithm_base import AlgorithmBase


class DQN(AlgorithmBase):
    """
    DQN algorithm implementation
    """

    def __init__(self, env: Any, target_network: NNBase, policy_net: NNBase,
                 n_max_iterations: int, tolerance: float, update_frequency: int,
                 batch_size: int, gamma: float, optimizer: Any, tau: float,
                 steps_per_iteration: int, state_size: int, n_actions: int,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995, device: str = 'cpu',
                 buffer_size: int = 100, seed: int = 0) -> None:

        super(DQN, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance, env=env)
        self.target_net = target_network
        self.policy_net = policy_net
        self.memory = ReplayBuffer(batch_size=batch_size, action_size=n_actions, device=device,
                                    buffer_size=buffer_size, seed=seed)
        self.optimizer = optimizer
        self._training_reward = 0
        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self._steps_per_iteration = steps_per_iteration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device
        self._state_size = state_size
        self.n_actions = n_actions
        self._scores = []
        self._scores_window = deque(maxlen=100)  # last 100 scores
        self.eps = eps_start

    @property
    def scores(self):
        return self._scores

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(DQN, self).actions_before_training_iterations(**options)
        self._scores = []

        # last 100 scores
        self._scores_window = deque(maxlen=100)
        self.eps = self.eps_start

    def actions_after_training_iterations(self, **options) -> None:
        pass

    """
    def act(self, state: Any, eps: float) -> Any:
        Returns actions for given state as per current policy.

        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self._local_net.eval()
        with torch.no_grad():
            action_values = self._local_net(state)
        self._local_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self._action_size))
    """

    """
    def learn(self, experiences: Any):
        
        Learn parameters from the given experience
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self._net(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self._local_net(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # ------------------- update target network ------------------- #
        soft_network_update(source=self._local_net, target=self._net, tau=self._tau)
    """

