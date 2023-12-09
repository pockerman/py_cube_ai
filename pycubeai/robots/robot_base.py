from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Union

class RobotBase(ABC):

    def __init__(self, name: str, specification: Union[Path, str, None]):
        self._name = name
        self._specification = specification
        self._is_initialized = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def specification(self) -> Union[Path, str, None]:
        return self._specification

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
