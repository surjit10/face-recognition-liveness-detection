import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_demo_overlay", PROJECT_ROOT / "scripts" / "run_demo.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_DEMO)


def _result(status, reason, state="RETRY", person_id=None):
    return SimpleNamespace(status=status, reason=reason, state=state, person_id=person_id)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_result("retry", "no_face"), "NO FACE"),
        (_result("retry", "multiple_faces"), "MULTIPLE FACES"),
        (_result("retry", "face_too_small"), "HOLD STILL"),
        (_result("retry", "collecting_pad_evidence", "COLLECTING_PAD"), "CHECKING LIVENESS"),
        (
            _result("spoof", "pad_below_threshold", "PAD_FAILED"),
            "LIVENESS NOT CONFIRMED",
        ),
        (_result("unknown", "pad_passed_identity_below_threshold", "UNKNOWN"), "UNKNOWN"),
        (_result("verified", "pad_passed_identity_accepted", "IDENTIFIED", "000"), "VERIFIED: 000"),
        (_result("verified", "pad_passed_identity_accepted", "IDENTIFIED"), "HOLD STILL"),
    ],
)
def test_overlay_messages_are_concise_and_fail_closed(result, expected) -> None:
    assert RUN_DEMO.overlay_text(result) == expected


@pytest.mark.parametrize("key", [ord("q"), 27])
def test_render_stops_on_quit_keys(monkeypatch, key) -> None:
    monkeypatch.setattr(RUN_DEMO.cv2, "putText", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUN_DEMO.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(RUN_DEMO.cv2, "waitKey", lambda delay: key)
    result = _result("verified", "pad_passed_identity_accepted", "IDENTIFIED", "000")

    _, stop = RUN_DEMO.render_and_report(SimpleNamespace(), result, 1, False, "IDENTIFIED")

    assert stop is True


class _Capture:
    def __init__(self) -> None:
        self.released = False

    def read(self):
        return False, None

    def release(self) -> None:
        self.released = True

    def isOpened(self) -> bool:
        return not self.released


def test_camera_read_failure_exits_two_and_releases(capsys) -> None:
    capture = _Capture()

    code = RUN_DEMO.run_camera(
        SimpleNamespace(), SimpleNamespace(frame_queue_size=2), capture, True
    )

    assert code == 2
    assert capture.released is True
    assert "camera read failed" in capsys.readouterr().err
