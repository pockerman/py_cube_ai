"""Module pytorch_optimizer_builder. Specifies
a simple factory for building PyTorch optimizers

"""

from typing import Any
import torch.optim as optim
from pycubeai.optimization.optimizer_type import OptimizerType

TORCH_OPTIMIZER_TYPES = [OptimizerType.ADAM]


def pytorch_optimizer_builder(opt_type: OptimizerType, model_params: Any, **options) -> optim.Optimizer:
    """ Factory method for building PyTorch optimizers

    Parameters
    ----------

    opt_type: The type of the optimizer
    model_params: Model parameters to optimize on
    options: Options for the optimizer

    Returns
    -------

    A concrete instance of the optim.Optimizer class
    """

    if opt_type not in TORCH_OPTIMIZER_TYPES:
        raise ValueError("Invalid PyTorch optimizer type. Type {0} not in {1}".format(opt_type.name,
                                                                                      TORCH_OPTIMIZER_TYPES))
    if opt_type == OptimizerType.ADAM:
        return optim.Adam(params=model_params, lr=options["learning_rate"])