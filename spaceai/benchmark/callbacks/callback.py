from typing import (
    Any,
    Dict,
)


class Callback:
    """Base class for implementing callbacks to be used during the execution of benchmarks."""

    def call(self):
        """Method to be called during the execution of the benchmark. It can be implemented
        to collect data, log information, compute metrics, etc."""
        raise NotImplementedError

    def __call__(self):
        """Method to be called during the execution of the benchmark. It calls the `call` method.
        It should not be overridden."""
        self.call()

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        """Method to collect data from the callback. It can be implemented to return the data
        collected during the execution of the benchmark. If `reset` is True, the callback should
        reset the data collected.

        Args:
            reset (bool): If True, the callback should reset the data collected.

        Returns:
            Dict[str, Any]: Data collected during the execution of the benchmark.
        """
        raise NotImplementedError
