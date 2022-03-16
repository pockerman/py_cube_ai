"""Module optimizer_type. Specifies an
enumeration for various PyTorch optimizers

"""
import enum


class OptimizerType(enum.IntEnum):

    INVALID = -1
    GD = 0
    SGD = 1
    BatchGD = 2
    ADAM = 3