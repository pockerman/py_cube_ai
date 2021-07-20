"""
Implementation of vanilla DQN
"""

from typing import Any
import torch.nn.functional as F

from networks.nn_base import NNBase
from networks.utils import soft_network_update
from utils.replay_buffer import ReplayBuffer
from algorithms.algorithm_base import AlgorithmBase


class DQN(AlgorithmBase):

    def __init__(self, env: Any, target_network: NNBase, local_net: NNBase,
                 n_max_iterations: int, tolerance: float, update_frequency: int,
                 batch_size: int, gamma: float, optimizer: Any, tau:float) -> None:
        super(DQN, self).__init__(n_max_iterations=n_max_iterations, tolerance=tolerance,env=env)
        self._net = target_network
        self._local_net = local_net
        self._memory = ReplayBuffer
        self._optimizer = optimizer
        self._training_reward = 0
        self._update_frequency = update_frequency
        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau


    @property
    def gamma(self) -> float:
        return self._gamma

    def step(self, **options) -> None:
        """
        One iteration step
        """

        # get an action
        action = self.act(state=self.state)

        # do one step in the environemnt
        next_state, reward, done, _ = self.train_env.step(action)

        # add into the memory
        self._memory.add(state=self.state, action=action, next_state=next_state,
                         reward=reward, done=done)

        if self.itr_control.current_itr_counter % self._update_frequency == 0:
            if len(self._memory) > self._batch_size:
                experiences = self._memory.sample()
                self.learn(experiences=experiences)

        self.state=next_state
        self._training_reward += reward

        if done:
            # update the residual so that we break
            self.itr_control.residual *= 10**-1


    def act(self, state: Any) -> Any:
        pass

    def learn(self, experiences: Any):
        """
        Learn parameters from the given experience
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """

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
        soft_network_update(net1=self._local_net, net2=self._net, tau=self._tau)


