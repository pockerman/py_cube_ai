"""The module play_info. Specifies the result
type when playing an agent on an environment

"""

from dataclasses import dataclass


@dataclass(init=True, repr=True)
class PlayInfo(object):
    pass