import enum

class OptimzerType(enum.IntEnum):

    INVALID = -1
    GD = 0
    SGD = 1
    BatchGD = 2
    ADAM = 3