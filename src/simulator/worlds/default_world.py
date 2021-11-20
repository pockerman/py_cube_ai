from src.simulator.worlds.world_base import WorldBase


class DefaultWorld(WorldBase):
    def __init__(self, period: float) -> None:
        super(DefaultWorld, self).__init__(period=period)