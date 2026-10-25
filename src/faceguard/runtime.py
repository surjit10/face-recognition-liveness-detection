"""Bounded latest-frame transport for live video inference."""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_bgr: np.ndarray
    captured_monotonic_ms: float
    sequence: int


class LatestFrameQueue:
    """A bounded queue that drops the oldest waiting frame when full."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize <= 0:
            raise ValueError("Frame queue maxsize must be positive")
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=maxsize)
        self.dropped_frames = 0

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def put_latest(self, packet: FramePacket) -> None:
        while True:
            try:
                self._queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                    self.dropped_frames += 1
                except queue.Empty:
                    continue

    def get(self, timeout: float | None = None) -> FramePacket:
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()
