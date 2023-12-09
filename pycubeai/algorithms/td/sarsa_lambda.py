from typing import TypeVar
from pycubeai.algorithms.td.td_algorithm_base import TDAlgoBase

AlgoInput = TypeVar('AlgoInput')

class SarsaLambda(TDAlgoBase):
    
    def __init__(self, input: AlgoInput):
        super(SarsaLambda, self).__init__()