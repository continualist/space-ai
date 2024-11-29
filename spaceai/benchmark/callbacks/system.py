import os
import psutil

from typing import Any, Dict

from .callback import Callback


class SystemMonitorCallback(Callback):
    """Callback to monitor the system resources used by the process."""

    def __init__(self):
        """Initialize the SystemMonitorCallback."""
        self.p: psutil.Process = psutil.Process(os.getpid())
        self.n: int = 0
        self.cpu: float = 0
        self.mem: float = 0

    def call(self):
        """Method to be executed by the callback. It collects the CPU and memory usage \
        of the process."""
        curr_cpu = self.p.cpu_percent() / psutil.cpu_count()
        curr_mem = self.p.memory_info().rss
        self.cpu = self.cpu + (curr_cpu - self.cpu) / (self.n + 1)
        self.mem = self.mem + (curr_mem - self.mem) / (self.n + 1)
        self.n += 1

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        """Collect data from the callback. It returns a dictionary with the data collected \
        by the callback.

        Args:
            reset (bool): If True, the callback should reset the data collected.

        Returns:
            Dict[str, Any]: Data collected by the callback.
        """
        data = {"cpu": self.cpu, "mem": self.mem}
        if reset:
            self.cpu = 0
            self.mem = 0
            self.n = 0
        return data
