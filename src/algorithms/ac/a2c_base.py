"""
a2c_base module. Implements a base class
for A2C like algorithms
"""

from torch import nn
from src.optimization.optimizer_type import OptimzerType

class A2CConfig(object):
    """
    Configuration parameters for A2C algorithms
    """
    def __init__(self):
        self.n_episodes: int = 0
        self.n_itrs_per_episode: int = 0
        self.opt_type: OptimzerType = OptimzerType.INVALID


class A2CBase(nn.Module):

    def __init__(self, config: A2CConfig) -> None:
        super(A2CBase, self).__init__()
        self.config = config

    def on_episode(self, **options):
        pass
