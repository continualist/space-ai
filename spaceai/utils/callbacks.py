import os
import time
from threading import Thread
from time import sleep
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import psutil


class Callback:
    def call(self):
        raise NotImplementedError

    def __call__(self):
        self.call()

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        raise NotImplementedError


class CallbackHandler:
    def __init__(self, callbacks: List[Callback], call_every_ms: int = 100):
        self.callbacks: List[Callback] = callbacks
        self.running: bool = False
        self.call_every_ms: int = call_every_ms
        self.thread: Optional[Thread] = None
        self.start_time: Optional[int] = None
        self.time: Optional[float] = None

    def start(self):
        self.thread = Thread(target=self.callback_loop)
        self.running = True
        self.thread.start()

    def callback_loop(self):
        self.start_time = time.time()
        while self.running:
            for callback in self.callbacks:
                callback()
            sleep(self.call_every_ms / 1000)

    def stop(self, blocking: bool = True):
        self.running = False
        end_time = time.time()
        self.time = end_time - self.start_time if self.start_time is not None else None
        if blocking and self.thread is not None:
            self.thread.join()

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        data = {"time": self.time}
        for callback in self.callbacks:
            data.update(callback.collect(reset=reset))
        return data


class SystemMonitorCallback(Callback):
    def __init__(self):
        self.p: psutil.Process = psutil.Process(os.getpid())
        self.n: int = 0
        self.cpu: float = 0
        self.mem: float = 0

    def call(self):
        curr_cpu = self.p.cpu_percent() / psutil.cpu_count()
        curr_mem = self.p.memory_info().rss
        self.cpu = self.cpu + (curr_cpu - self.cpu) / (self.n + 1)
        self.mem = self.mem + (curr_mem - self.mem) / (self.n + 1)
        self.n += 1

    def collect(self, reset: bool = False) -> Dict[str, Any]:
        data = {"cpu": self.cpu, "mem": self.mem}
        if reset:
            self.cpu = 0
            self.mem = 0
            self.n = 0
        return data
