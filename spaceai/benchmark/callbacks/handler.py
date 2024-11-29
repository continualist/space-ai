from __future__ import annotations

import time
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from .callback import Callback


class CallbackHandler:
    """Class to handle the execution of a list of callbacks during the execution of a
    benchmark."""

    def __init__(self, callbacks: List[Callback], call_every_ms: int = 100):
        """Initialize the CallbackHandler.

        Args:
            callbacks (List[Callback]): List of callbacks to be executed during the benchmark.
            call_every_ms (int): Time in milliseconds to call the callbacks.
        """

        self.callbacks: List[Callback] = callbacks
        self.running: bool = False
        self.call_every_ms: int = call_every_ms
        self.thread: Optional[Thread] = None
        self.start_time: Optional[int] = None
        self.time: Optional[float] = None

    def start(self):
        """Start the execution of the handler's thread."""
        self.thread = Thread(target=self.callback_loop)
        self.running = True
        self.thread.start()

    def callback_loop(self):
        """Method to be executed by the handler's thread.

        It activates the callbacks at the specified time interval.
        """
        self.start_time = time.time()
        while self.running:
            for callback in self.callbacks:
                callback()
            time.sleep(self.call_every_ms / 1000)

    def stop(self, blocking: bool = True):
        """Stop the execution of the handler's thread.

        Args:
            blocking (bool): If True, the method blocks until the thread is stopped.
        """
        self.running = False
        end_time = time.time()
        self.time = end_time - self.start_time if self.start_time is not None else None
        if blocking and self.thread is not None:
            self.thread.join()

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        """Collect data from the callbacks. It collects data from all the callbacks and
        returns a dictionary with the data collected.

        Args:
            reset (bool): If True, the callbacks should reset the data collected.

        Returns:
            Dict[str, Any]: Data collected from the callbacks.
        """
        data = {"time": self.time}
        for callback in self.callbacks:
            data.update(callback.collect(reset=reset))
        return data
