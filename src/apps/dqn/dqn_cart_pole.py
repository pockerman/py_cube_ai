"""
Cart-pole problem using DQN. The implementation
is taken from the  PyTorch documentation here
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import numpy as np
import gym
import gym.wrappers as wrappers
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from src.algorithms import CartPoleDQN
from src.networks.nn_base import NNBase


class Network(NNBase):

    def __init__(self, h, w, outputs, device: str):
        super(Network, self).__init__(device_type=device)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # helper function
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, state):
        state = state.to(self.device)
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        return self.head(state.view(state.size(0), -1))


class TargetNet(Network):
    """
    The target network
    """
    def __init__(self, h, w, outputs, device: str):
        super(TargetNet, self).__init__(h=h, w=w, outputs=outputs, device=device)


class QNet(Network):
    """
    Q-network
    """

    def __init__(self, h, w, outputs, device: str):
        super(QNet, self).__init__(h=h, w=w, outputs=outputs, device=device)


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    """
    Returns the cart location given the screen width
    :param screen_width:
    :return:
    """
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    """
    Helper function for obtaining screen data.
    Returned screen requested by gym is 400x600x3, but is sometimes larger
    such as 800x1200x3. Transpose it into torch order (CHW)
    """

    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
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
    return resize(screen).unsqueeze(0)


if __name__ == '__main__':

    # various constants
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    NUM_EPISODES = 50
    SEED = 42
    BUFFER_SIZE = 10000
    ENV_NAME = 'CartPole-v0'

    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    print("Start training DQN on {}".format(ENV_NAME))
    env = wrappers.Monitor(gym.make(ENV_NAME).unwrapped, './cart_pole_movie/', force=True)
    env.reset()

    action_size = env.action_space.n

    init_screen = get_screen(env=env)
    _, _, screen_height, screen_width = init_screen.shape

    target_net = TargetNet(screen_height, screen_width, action_size, device=device)
    policy_net = QNet(screen_height, screen_width, action_size, device=device)

    # the optimizer to use
    optimizer = optim.RMSprop(policy_net.parameters())

    agent = CartPoleDQN(env=env, target_network=target_net, policy_net=policy_net,
                 n_max_iterations=NUM_EPISODES, tolerance=1.0e-8, update_frequency=TARGET_UPDATE,
                 batch_size=BATCH_SIZE, gamma=GAMMA, optimizer=optimizer, tau=0.4,
                 steps_per_iteration=1000, state_size=10, n_actions=env.action_space.n,
                 eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, device=device,
                 buffer_size=BUFFER_SIZE, seed=SEED)

    # Train the agent
    agent.train()

    print("Finished training DQN on {}".format(ENV_NAME))
    env.close()
