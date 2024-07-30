# pylint: disable=missing-module-docstring
from .benchmark import SpaceBenchmark


class NASABenchmark(
    SpaceBenchmark
):  # pylint: disable=missing-class-docstring, too-few-public-methods

    def __init__(self):  # pylint: disable=useless-parent-delegation
        super().__init__()
