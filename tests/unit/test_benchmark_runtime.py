from pathlib import Path

import pytest

from scripts.benchmark_runtime import _performance_gate


def _objective(path: Path, status: str = "FROZEN") -> Path:
    path.write_text(
        f"""schema_version: 1
status: {status}
objectives:
  minimum_processed_fps: 5.0
  maximum_p95_capture_to_decision_ms: 500.0
  maximum_peak_memory_mb: 750.0
  maximum_stable_face_verification_seconds: 3.0
  maximum_read_failures: 0
  camera_released_on_shutdown: true
  bounded_latest_frame_queue_size: 2
""",
        encoding="utf-8",
    )
    return path


def test_performance_gate_passes_all_frozen_objectives(tmp_path: Path) -> None:
    objective = _objective(tmp_path / "objective.yaml")
    session = {
        "processed_fps": 6.25,
        "p95_latency_ms": 300.0,
        "peak_memory_mb": 650.0,
        "stable_face_verification_ms": 1250.0,
        "read_failures": False,
        "camera_released": True,
    }

    result = _performance_gate(session, objective, queue_size=2)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"].values())
    assert result["checks"]["stable_face_verification_seconds"]["observed"] == 1.25


def test_performance_gate_rejects_unfrozen_objective(tmp_path: Path) -> None:
    objective = _objective(tmp_path / "objective.yaml", status="PROPOSED")

    with pytest.raises(RuntimeError, match="must be FROZEN"):
        _performance_gate({}, objective, queue_size=2)
