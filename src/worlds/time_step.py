from collections import namedtuple

TimeStep = namedtuple("TimeStep", ["state", "reward", "done", "info"])