"""
DQN algorithm applied on the gym cart pole
environment. The implementation is taken from PyTorch tutorial
from here https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
from typing import Any
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from pycubeai.networks_org.nn_base import NNBase
from pycubeai.utils.replay_buffer import ExperienceTuple
from .dqn import DQN


class CartPoleDQN(DQN):
    """
    The CartPoleDQN class handles the implementation
    of DQN on the CartPole environment
    """

    resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    def __init__(self, env, target_network: NNBase, policy_net: NNBase,
                 n_episodes: int, tolerance: float, update_frequency: int,
                 batch_size: int, gamma: float, optimizer: Any, tau: float,
                 steps_per_iteration: int, state_size: int, n_actions: int,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.995, device: str = 'cpu',
                 buffer_size: int = 100, seed: int = 0
                 ):
        super(CartPoleDQN, self).__init__(env=env, target_network=target_network, policy_net=policy_net,
                                          n_episodes=n_episodes, tolerance=tolerance, update_frequency=update_frequency,
                                          batch_size=batch_size, gamma=gamma, optimizer=optimizer, tau=tau,
                                          steps_per_iteration=steps_per_iteration, state_size=state_size, n_actions=n_actions,
                                          eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device, buffer_size=buffer_size, seed=seed)

        self._last_screen = None
        self._current_screen = None
        self._steps_done = 0

    def actions_before_training_begins(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(CartPoleDQN, self).actions_before_training_begins(**options)
        self._last_screen = self.get_screen()
        self._current_screen = self.get_screen()
        self._steps_done = 0

    def on_episode(self, **options) -> None:
        """
        One iteration step
        """

        # reset the environment
        self.train_env.reset()
        self._last_screen = self.get_screen()
        self._current_screen = self.get_screen()
        self.state = self._current_screen - self._last_screen

        score = 0
        for itr in range(self._steps_per_iteration):

            action = self.select_action(self.state)

            # do one step in the environemnt
            _, reward, done, _ = self.train_env.on_episode(action.item())

            self._training_reward += reward
            score += reward

            reward = torch.tensor([reward], device=self.device)

            # Observe new state
            self._last_screen = self._current_screen
            print(self._last_screen.shape)
            self._current_screen = self.get_screen()

            if not done:
                next_state = self._current_screen - self._last_screen
            else:
                next_state = None

            # add into the memory
            self.memory.add(state=self.state,
                            action=action,
                            next_state=next_state,
                            reward=reward,
                            done=done)

            # update the state
            self.state = next_state
            self._scores_window.append(score)  # save most recent score
            self._scores.append(score)  # save most recent score

            # optimize the model
            self.optimize_model()

            # decrease epsilon
            self.eps = max(self.eps_end, self.eps_decay * self.eps)
            print('\tAverage Score: {:.2f}\n'.format(np.mean(self._scores_window)), end="")

            if done:
                break

            if self.itr_control.current_itr_counter % self.update_frequency == 0:
                # update the target network
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.itr_control.current_itr_counter % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.itr_control.current_itr_counter,
                                                                   np.mean(self._scores_window)))
            #if np.mean(self._scores_window) >= 200.0:
            #    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(self.itr_control.current_itr_counter - 100,
            #                                                                                         np.mean(self._scores_window)))
                #torch.save(self._local_net.state_dict(), 'checkpoint.pth')

            #    break
                # update the residual so that we break
                #self.itr_control.residual *= 10**-1

    def select_action(self, state):
        """
        Select an action given the state
        """

        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self._steps_done / self.eps_decay)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        try:
            transitions = self.memory.sample()
            batch = ExperienceTuple(*zip(*transitions))
        except ValueError as e:
            print("Exception is thrown ", str(e))

        # Compute a mask of non - final states and concatenate
        # the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.on_episode()

    def get_screen(self):
        """
        Helper function for obtaining screen data.
        Returned screen requested by gym is 400x600x3, but is sometimes larger
        such as 800x1200x3. Transpose it into torch order (CHW)
        """

        screen = self.train_env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return CartPoleDQN.resize(screen).unsqueeze(0)

    def get_cart_location(self, screen_width):
        """
        Returns the cart location given the screen width
        :param screen_width:
        :return:
        """
        world_width = self.train_env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.train_env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

