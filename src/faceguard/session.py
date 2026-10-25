"""Threaded, bounded live-camera orchestration for desktop and service front ends."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import cv2
import numpy as np

from .pipeline import VerificationResult, VideoVerifier
from .presentation import safe_event
from .runtime import FramePacket, LatestFrameQueue
from .runtime_config import RuntimeConfig
from .telemetry import RollingMetrics


READ_FAILURE_LIMIT = 3


class SessionStatus(str, Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class SessionOptions:
    camera_index: int = 0
    max_reconnect_attempts: int = 3
    reconnect_delay_s: float = 1.0

    def validate(self) -> None:
        if self.camera_index != 0:
            raise ValueError("Only authorized camera 0 is supported")
        if self.max_reconnect_attempts < 0:
            raise ValueError("Reconnect attempts must not be negative")
        if self.reconnect_delay_s < 0:
            raise ValueError("Reconnect delay must not be negative")


@dataclass(frozen=True)
class SessionMetrics:
    fps: float
    median_age_ms: float
    p95_age_ms: float
    median_latency_ms: float
    processed: int
    captured: int
    dropped: int
    elapsed_s: float


@dataclass(frozen=True)
class UiSnapshot:
    """Newest presentation state; the frame is memory-only and never serializable."""

    sequence: int
    frame_bgr: np.ndarray
    result: VerificationResult
    metrics: SessionMetrics


@dataclass(frozen=True)
class SessionSummary:
    status: SessionStatus
    captured: int
    processed: int
    dropped: int
    camera_released: bool
    error: str | None


CaptureFactory = Callable[[int], Any]
EventSink = Callable[[dict[str, object]], None]


class VerificationSession:
    """Own camera capture and inference without allowing unbounded frame buildup."""

    def __init__(
        self,
        *,
        verifier: VideoVerifier,
        config: RuntimeConfig,
        options: SessionOptions | None = None,
        capture_factory: CaptureFactory = cv2.VideoCapture,
        event_sink: EventSink | None = None,
    ) -> None:
        self.verifier = verifier
        self.config = config
        self.options = options or SessionOptions()
        self.options.validate()
        self._capture_factory = capture_factory
        self._event_sink = event_sink
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._verifier_lock = threading.Lock()
        self._status = SessionStatus.IDLE
        self._error: str | None = None
        self._capture: Any | None = None
        self._capture_released = True
        self._closed = False
        self._captured = 0
        self._last_event_state: object | None = None
        self._producer: threading.Thread | None = None
        self._consumer: threading.Thread | None = None
        self._initialize_transports()

    def _initialize_transports(self) -> None:
        self._frames = LatestFrameQueue(maxsize=self.config.frame_queue_size)
        self._updates: queue.Queue[UiSnapshot] = queue.Queue(maxsize=1)
        self._stopping = threading.Event()
        self._producer_done = threading.Event()
        self._metrics = RollingMetrics()

    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def is_running(self) -> bool:
        return self.status in {
            SessionStatus.STARTING,
            SessionStatus.RUNNING,
            SessionStatus.RECONNECTING,
            SessionStatus.STOPPING,
        }

    @property
    def threads_alive(self) -> bool:
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._producer, self._consumer)
        )

    def _set_status(self, status: SessionStatus, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            if error is not None:
                self._error = error

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Verification session is closed")
        if self.threads_alive or self.is_running:
            raise RuntimeError("Verification session is already running")
        self.options.validate()
        with self._verifier_lock:
            self.verifier.reset()
        self._initialize_transports()
        with self._lock:
            self._captured = 0
            self._error = None
            self._capture_released = True
            self._last_event_state = None
            self._status = SessionStatus.STARTING
        self._producer = threading.Thread(
            target=self._capture_loop,
            name="faceguard-app-capture",
            daemon=True,
        )
        self._consumer = threading.Thread(
            target=self._inference_loop,
            name="faceguard-app-inference",
            daemon=True,
        )
        self._producer.start()
        self._consumer.start()

    def _replace_capture(self, capture: Any | None) -> None:
        with self._capture_lock:
            self._capture = capture
            self._capture_released = capture is None or not capture.isOpened()

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture = self._capture
            if capture is not None:
                capture.release()
            self._capture_released = True

    def _open_capture(self) -> Any | None:
        capture = self._capture_factory(self.options.camera_index)
        if not capture.isOpened():
            capture.release()
            return None
        self._replace_capture(capture)
        return capture

    def _capture_loop(self) -> None:
        sequence = 0
        failures = 0
        capture: Any | None = None
        try:
            capture = self._open_capture()
            if capture is None:
                self._set_status(SessionStatus.FAILED, "Unable to open authorized camera 0")
                self._stopping.set()
                return
            self._set_status(SessionStatus.RUNNING)
            while not self._stopping.is_set():
                ok, frame = capture.read()
                if ok and isinstance(frame, np.ndarray):
                    failures = 0
                    sequence += 1
                    with self._lock:
                        self._captured = sequence
                    self._frames.put_latest(
                        FramePacket(frame, time.monotonic() * 1000.0, sequence)
                    )
                    continue

                failures += 1
                if failures < READ_FAILURE_LIMIT:
                    self._stopping.wait(0.05)
                    continue

                with self._verifier_lock:
                    self.verifier.reset()
                self._set_status(SessionStatus.RECONNECTING)
                self._release_capture()
                capture = None
                for _ in range(self.options.max_reconnect_attempts):
                    if self._stopping.wait(self.options.reconnect_delay_s):
                        break
                    capture = self._open_capture()
                    if capture is not None:
                        failures = 0
                        with self._verifier_lock:
                            self.verifier.reset()
                        self._set_status(SessionStatus.RUNNING)
                        break
                if capture is None:
                    if not self._stopping.is_set():
                        self._set_status(
                            SessionStatus.FAILED,
                            "Camera connection was lost and could not be restored",
                        )
                        self._stopping.set()
                    break
        except Exception as exc:
            self._set_status(
                SessionStatus.FAILED,
                f"Camera session failed safely: {type(exc).__name__}",
            )
            self._stopping.set()
        finally:
            self._release_capture()
            self._producer_done.set()

    def _inference_loop(self) -> None:
        try:
            while not self._producer_done.is_set() or self._frames.size:
                try:
                    packet = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    with self._verifier_lock:
                        result = self.verifier.process(
                            packet.frame_bgr,
                            packet.captured_monotonic_ms,
                        )
                    age_ms = max(
                        0.0,
                        time.monotonic() * 1000.0 - packet.captured_monotonic_ms,
                    )
                    self._metrics.add(result.latency_ms, age_ms)
                    snapshot = UiSnapshot(
                        sequence=packet.sequence,
                        frame_bgr=packet.frame_bgr,
                        result=result,
                        metrics=self._metrics_snapshot(),
                    )
                    self._put_latest_snapshot(snapshot)
                    state = getattr(result, "state", None)
                    if self._event_sink is not None and state != self._last_event_state:
                        self._event_sink(safe_event(packet.sequence, result))
                        self._last_event_state = state
                finally:
                    self._frames.task_done()
        except Exception as exc:
            self._set_status(
                SessionStatus.FAILED,
                f"Analysis stopped safely: {type(exc).__name__}",
            )
            self._stopping.set()
            self._release_capture()
        finally:
            with self._verifier_lock:
                self.verifier.reset()
            if self.status not in {SessionStatus.FAILED, SessionStatus.STOPPING}:
                self._set_status(SessionStatus.STOPPED)
            elif self.status == SessionStatus.STOPPING:
                self._set_status(SessionStatus.STOPPED)

    def _metrics_snapshot(self) -> SessionMetrics:
        with self._lock:
            captured = self._captured
        return SessionMetrics(
            fps=self._metrics.fps(),
            median_age_ms=self._metrics.median_age(),
            p95_age_ms=self._metrics.p95_age(),
            median_latency_ms=self._metrics.median_latency(),
            processed=self._metrics.processed,
            captured=captured,
            dropped=self._frames.dropped_frames,
            elapsed_s=self._metrics.elapsed_s,
        )

    def _put_latest_snapshot(self, snapshot: UiSnapshot) -> None:
        while True:
            try:
                self._updates.put_nowait(snapshot)
                return
            except queue.Full:
                try:
                    self._updates.get_nowait()
                except queue.Empty:
                    continue

    def latest_snapshot(self) -> UiSnapshot | None:
        latest: UiSnapshot | None = None
        while True:
            try:
                latest = self._updates.get_nowait()
            except queue.Empty:
                return latest

    def stop(self, *, wait: bool = True, timeout: float = 3.0) -> None:
        current = self.status
        if current in {SessionStatus.IDLE, SessionStatus.STOPPED} and not self.threads_alive:
            with self._verifier_lock:
                self.verifier.reset()
            return
        if current != SessionStatus.FAILED:
            self._set_status(SessionStatus.STOPPING)
        self._stopping.set()
        self._release_capture()
        if wait:
            self.join(timeout=timeout)
        if not self.threads_alive and self.status != SessionStatus.FAILED:
            self._set_status(SessionStatus.STOPPED)
        if not self.threads_alive:
            with self._verifier_lock:
                self.verifier.reset()

    def join(self, *, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in (self._producer, self._consumer):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        if self._closed:
            return
        self.stop(wait=True)
        with self._verifier_lock:
            self.verifier.end()
        self._closed = True

    def summary(self) -> SessionSummary:
        metrics = self._metrics_snapshot()
        with self._lock:
            released = self._capture_released
        return SessionSummary(
            status=self.status,
            captured=metrics.captured,
            processed=metrics.processed,
            dropped=metrics.dropped,
            camera_released=released,
            error=self.error,
        )
