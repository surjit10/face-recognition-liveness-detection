import http.client
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from faceguard.live_metrics import (
    LiveMetricsHub,
    LiveMetricsServer,
    load_performance_targets,
    prometheus_text,
)
from faceguard.pipeline import VerificationResult, VerificationState
from faceguard.session import VerificationSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _result(state=VerificationState.IDENTIFIED, reason="pad_passed_identity_accepted",
            latency=50.0, person_id="0042"):
    return VerificationResult(
        status="verified" if state == VerificationState.IDENTIFIED else "retry",
        state=state,
        person_id=person_id,
        similarity=0.8123,
        pad_score=0.9731,
        track_id=7,
        model_version="test",
        reason=reason,
        latency_ms=latency,
        pad_observations=8,
    )


def test_snapshot_aggregates_window_and_never_exposes_identity() -> None:
    clock = _Clock()
    hub = LiveMetricsHub(source="camera", window=10, clock=clock)
    for index in range(20):
        clock.now += 0.1
        hub.record(_result(latency=10.0 * (index + 1)), age_ms=20.0 + index,
                   captured=index + 1, dropped=index // 2, queue_depth=1)

    snap = hub.snapshot()

    assert snap["frames"] == {"processed": 20, "captured": 20, "dropped": 9,
                              "drop_ratio": 0.45, "queue_depth": 1}
    assert snap["window"]["frames"] == 10
    assert snap["window"]["fps"] == pytest.approx(10.0)
    assert snap["window"]["latency_ms_p95"] == 200.0  # last 10 latencies: 110..200
    assert snap["window"]["latency_ms_p50"] == 150.0
    assert snap["pipeline"]["state_counts"] == {"IDENTIFIED": 20}
    assert len(snap["history"]) == 20
    rendered = json.dumps(snap)
    for secret in ("0042", "0.8123", "0.9731", "person_id", "similarity", "track"):
        assert secret not in rendered


def test_history_keeps_two_minutes_and_reset_clears_everything() -> None:
    clock = _Clock()
    hub = LiveMetricsHub(source="video", clock=clock)
    for _ in range(5):
        hub.record(_result(), age_ms=1.0, captured=1, dropped=0)
        clock.now += 60.0
    assert [round(p[0]) for p in hub.snapshot()["history"]] == [120, 180, 240]

    hub.reset()
    snap = hub.snapshot()
    assert snap["frames"]["processed"] == 0
    assert snap["history"] == [] and snap["pipeline"]["state_counts"] == {}


def test_unexpected_reason_text_is_replaced_and_events_are_codes_only() -> None:
    hub = LiveMetricsHub(source="camera")
    hub.record(_result(reason="Welcome, Alice!"), age_ms=1.0, captured=1, dropped=0)
    hub.note_event("camera_reconnected")

    snap = hub.snapshot()
    assert snap["pipeline"]["reason_counts"] == {"other": 1}
    assert snap["events"] == {"camera_reconnected": 1}
    with pytest.raises(ValueError):
        hub.note_event("Camera lost for Alice")


def test_prometheus_text_has_typed_labelled_series() -> None:
    hub = LiveMetricsHub(source="camera")
    hub.record(_result(), age_ms=12.0, captured=3, dropped=1)
    text = prometheus_text(hub.snapshot())

    assert "# TYPE faceguard_frames_dropped_total counter" in text
    assert 'faceguard_frames_dropped_total{source="camera"} 1' in text
    assert 'faceguard_latency_ms{quantile="0.95",source="camera"} 50' in text
    assert 'faceguard_pipeline_state_total{source="camera",state="IDENTIFIED"} 1' in text
    assert text.endswith("\n")


def test_targets_come_from_frozen_performance_objective() -> None:
    targets = load_performance_targets(PROJECT_ROOT / "configs/performance_objective.yaml")
    assert targets == {"min_fps": 5.0, "max_p95_frame_age_ms": 500.0,
                       "max_peak_memory_mb": 750.0}


def test_server_refuses_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LiveMetricsServer(LiveMetricsHub(source="camera"), host="0.0.0.0", port=0)


def _get(port: int, path: str, host: str | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    headers = {"Host": host} if host else {}
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response, body


def test_server_serves_json_prometheus_and_dashboard_same_origin(tmp_path: Path) -> None:
    page = tmp_path / "dash.html"
    page.write_text("<html>dash</html>", encoding="utf-8")
    hub = LiveMetricsHub(source="camera")
    hub.record(_result(), age_ms=5.0, captured=1, dropped=0)
    server = LiveMetricsServer(hub, port=0, dashboard=page).start()
    port = int(server.url.rsplit(":", 1)[1].strip("/"))
    try:
        response, body = _get(port, "/metrics.json")
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/json"
        assert response.getheader("Cache-Control") == "no-store"
        assert response.getheader("Access-Control-Allow-Origin") is None
        assert json.loads(body)["frames"]["processed"] == 1

        response, body = _get(port, "/metrics")
        assert response.status == 200 and b"faceguard_processed_fps" in body

        response, body = _get(port, "/")
        assert response.status == 200 and body == b"<html>dash</html>"

        response, _ = _get(port, "/reports/../configs/runtime.yaml")
        assert response.status == 404
    finally:
        server.close()


def test_server_rejects_foreign_host_header_dns_rebinding() -> None:
    server = LiveMetricsServer(LiveMetricsHub(source="camera"), port=0).start()
    port = int(server.url.rsplit(":", 1)[1].strip("/"))
    try:
        response, _ = _get(port, "/metrics.json", host=f"evil.example:{port}")
        assert response.status == 403
        response, _ = _get(port, "/metrics.json", host=f"localhost:{port}")
        assert response.status == 200
    finally:
        server.close()


class _Capture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        time.sleep(0.003)
        return (False, None) if self.released else (True, np.zeros((8, 8, 3), np.uint8))

    def release(self) -> None:
        self.released = True


class _Verifier:
    def reset(self) -> None: ...

    def end(self) -> None: ...

    def process(self, frame, timestamp_ms):
        return _result(state=VerificationState.RETRY, reason="no_face", latency=2.0)


def test_desktop_session_feeds_hub_and_resets_it_per_session() -> None:
    hub = LiveMetricsHub(source="desktop")
    hub.record(_result(), age_ms=1.0, captured=999, dropped=999)  # stale previous session
    session = VerificationSession(
        verifier=_Verifier(),  # type: ignore[arg-type]
        config=SimpleNamespace(frame_queue_size=2),  # type: ignore[arg-type]
        capture_factory=lambda _: _Capture(),
        metrics_hub=hub,
    )
    session.start()
    deadline = time.monotonic() + 2.0
    while hub.snapshot()["frames"]["processed"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    session.stop(wait=True)

    snap = hub.snapshot()
    assert snap["frames"]["processed"] >= 3
    assert snap["frames"]["captured"] < 999
    assert set(snap["pipeline"]["state_counts"]) == {"RETRY"}
