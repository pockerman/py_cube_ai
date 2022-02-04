from typing import Any
import torch.optim as optim
from src.utils.optimizer_type import OptimzerType

TORCH_OPTIMIZER_TYPES = [OptimzerType.ADAM]


def build_optimizer(opt_type: OptimzerType, model_params: Any, **options):

    if opt_type not in TORCH_OPTIMIZER_TYPES:
        raise ValueError("Invalid PyTorch optimizer type. Type {0} not in {1}".format(opt_type.name,
                                                                                      TORCH_OPTIMIZER_TYPES))
    if opt_type == OptimzerType.ADAM:
        return optim.Adam(params=model_params, lr=options["learning_rate"])