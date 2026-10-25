import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_demo_reconnect", PROJECT_ROOT / "scripts" / "run_demo.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_DEMO)


class MockCapture:
    def __init__(self, reads=None):
        self.reads = list(reads or [])
        self.released = False
        self.opened = True

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True
        self.opened = False

    def isOpened(self):
        return self.opened


class MockVerifier:
    def __init__(self):
        self.reset_count = 0
        self.processed = 0

    def reset(self):
        self.reset_count += 1

    def process(self, frame, timestamp_ms):
        self.processed += 1
        return SimpleNamespace(
            status="retry",
            state=SimpleNamespace(value="RETRY"),
            reason="no_face",
            person_id=None,
            latency_ms=1.0,
            event=lambda: {"event": "test"},
            pad_observations=0,
        )


def test_reconnect_succeeds_after_temporary_failure(monkeypatch):
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    initial_capture = MockCapture([(True, dummy_frame), (False, None), (False, None), (False, None)])
    reconnected_capture = MockCapture([(True, dummy_frame), (True, dummy_frame)])

    created_captures = []

    def mock_video_capture(source):
        created_captures.append(source)
        return reconnected_capture

    monkeypatch.setattr(RUN_DEMO.cv2, "VideoCapture", mock_video_capture)
    monkeypatch.setattr(RUN_DEMO, "render_and_report", lambda *args, **kwargs: (None, len(created_captures) > 0))

    verifier = MockVerifier()
    config = SimpleNamespace(
        frame_queue_size=2,
        pipeline_policy=SimpleNamespace(pad_min_observations=8),
    )

    code = RUN_DEMO.run_camera(
        verifier,
        config,
        initial_capture,
        no_display=True,
        camera_source=0,
        max_reconnect_attempts=2,
        reconnect_delay_s=0.01,
    )

    assert code == 0
    assert initial_capture.released is True
    assert len(created_captures) == 1
    assert created_captures[0] == 0
    assert verifier.reset_count >= 1


def test_reconnect_exhausted_exits_with_code_2(monkeypatch):
    initial_capture = MockCapture([(False, None), (False, None), (False, None)])
    failed_new_cap = MockCapture([(False, None)])

    monkeypatch.setattr(RUN_DEMO.cv2, "VideoCapture", lambda source: failed_new_cap)

    verifier = MockVerifier()
    config = SimpleNamespace(
        frame_queue_size=2,
        pipeline_policy=SimpleNamespace(pad_min_observations=8),
    )

    code = RUN_DEMO.run_camera(
        verifier,
        config,
        initial_capture,
        no_display=True,
        camera_source=0,
        max_reconnect_attempts=2,
        reconnect_delay_s=0.01,
    )

    assert code == 2
    assert initial_capture.released is True
    assert verifier.reset_count >= 1


def test_video_file_normal_eof_does_not_reconnect():
    capture = MockCapture([(True, np.zeros((10, 10, 3), dtype=np.uint8)), (False, None)])
    capture.get = lambda prop: 30.0

    verifier = MockVerifier()
    code = RUN_DEMO.run_video(
        verifier,
        capture,
        no_display=True,
        pad_required=8,
    )

    assert code == 0
    assert verifier.processed == 1
