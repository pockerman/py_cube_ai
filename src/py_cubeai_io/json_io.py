"""The json_io module. Provides various
utilities for reading and writing state value functions,
state-action value functions e.t.c from/to json format

"""
from typing import TypeVar
import json
from pathlib import Path


QTable = TypeVar('QTable')


def write_q_function(qtable: QTable, filename: Path, **options) -> None:
    """
    Write the Q-function on the given file

    Parameters
    ----------
    qtable: The qtable to write
    filename: Path to the file to write to
    options: Any options needed

    Returns
    -------

    None

    """
    with open(filename, 'w') as fh:
        new_table = {}
        for key in qtable:
            new_table[str(key)] = qtable[key]
        json.dump(new_table, fh, indent=4)


def load_q_function(filename: Path, **options) -> dict:
    """
    Load the Q-function from the given json file

    Parameters
    ----------
    filename: Path to the json file
    options: Any options passed by the client

    Returns
    -------

    An instance of dict representing the q_table

    """
    with open(filename, 'r') as fh:
        q_table = json.load(fh)
        return q_table

