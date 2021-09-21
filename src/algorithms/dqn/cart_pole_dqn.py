"""
DQN algorithm applied on the gym cart pole
environment
"""
import numpy as np
from .dqn import DQN
from src.networks.nn_base import NNBase


class CartPoleDQN(DQN):
    """
    The CartPoleDQN class handles the implementation
    of DQN on the CartPole environment
    """

    def __init__(self, env, target_network: NNBase, policy_net: NNBase):
        super(CartPoleDQN, self).__init__(env=env, target_network=target_network, policy_net=policy_net)

        last_screen = None
        current_screen = None

    def actions_before_training_iterations(self, **options) -> None:
        """
        Execute any actions the algorithm needs before
        starting the iterations
        """
        super(CartPoleDQN, self).actions_before_training_iterations(**options)

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

