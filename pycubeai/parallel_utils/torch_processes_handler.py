"""
Small utility class to handle process
"""

from typing import Any
import torch.multiprocessing as mp


class TorchProcsHandler(object):

    def __init__(self, n_procs: int) -> None:
        self.n_procs = n_procs
        self.processes = []

    def create_and_start(self, target: Any, *args) -> None:
        for i in range(self.n_procs):
            p = mp.Process(target=target, args=args)
            p.start()
            self.processes.append(p)

    def join(self) -> None:
        for p in self.processes:
            p.join()

    def terminate(self) -> None:
        for p in self.processes:
            p.terminate()

    def join_and_terminate(self):
        self.join()
        self.terminate()
