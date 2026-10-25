import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from faceguard.pipeline import VerificationResult, VerificationState
from faceguard.session import SessionOptions, SessionStatus, VerificationSession


class _StreamingCapture:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.frame = np.zeros((24, 32, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self):
        if self.released or not self.opened:
            return False, None
        time.sleep(0.003)
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


class _Verifier:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.end_calls = 0
        self.process_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def process(self, frame, timestamp_ms):
        self.process_calls += 1
        return VerificationResult(
            status="retry",
            state=VerificationState.RETRY,
            person_id=None,
            similarity=None,
            pad_score=None,
            track_id=None,
            model_version="test",
            reason="no_face",
            latency_ms=1.0,
            pad_observations=0,
        )

    def end(self):
        self.end_calls += 1


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_session_restricts_live_source_to_camera_zero() -> None:
    with pytest.raises(ValueError, match="camera 0"):
        SessionOptions(camera_index=1).validate()


def test_session_processes_latest_frames_and_releases_camera() -> None:
    capture = _StreamingCapture()
    verifier = _Verifier()
    events = []
    session = VerificationSession(
        verifier=verifier,  # type: ignore[arg-type]
        config=SimpleNamespace(frame_queue_size=2),  # type: ignore[arg-type]
        capture_factory=lambda _: capture,
        event_sink=events.append,
    )

    session.start()
    assert _wait_until(lambda: session.latest_snapshot() is not None)
    session.stop(wait=True)

    summary = session.summary()
    assert summary.status == SessionStatus.STOPPED
    assert summary.camera_released is True
    assert summary.processed > 0
    assert capture.released is True
    assert session.threads_alive is False
    assert verifier.reset_calls >= 2
    assert len(events) == 1
    assert "frame" not in events[0]


def test_session_fails_closed_when_camera_cannot_open() -> None:
    capture = _StreamingCapture(opened=False)
    session = VerificationSession(
        verifier=_Verifier(),  # type: ignore[arg-type]
        config=SimpleNamespace(frame_queue_size=2),  # type: ignore[arg-type]
        capture_factory=lambda _: capture,
    )

    session.start()
    assert _wait_until(lambda: session.status == SessionStatus.FAILED)
    session.join(timeout=1.0)

    assert session.summary().camera_released is True
    assert "Unable to open" in (session.error or "")
    assert capture.released is True


def test_close_is_idempotent_and_ends_verifier() -> None:
    verifier = _Verifier()
    session = VerificationSession(
        verifier=verifier,  # type: ignore[arg-type]
        config=SimpleNamespace(frame_queue_size=2),  # type: ignore[arg-type]
        capture_factory=lambda _: _StreamingCapture(opened=False),
    )

    session.close()
    session.close()

    assert verifier.end_calls == 1
    assert not any(
        thread.name.startswith("faceguard-app-")
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
    )
