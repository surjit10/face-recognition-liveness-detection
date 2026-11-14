import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from faceguard.pipeline import PipelinePolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "apply_recognition_quality_gate",
    PROJECT_ROOT / "scripts" / "apply_recognition_quality_gate.py",
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _policy() -> PipelinePolicy:
    return PipelinePolicy(
        pad_threshold=0.7,
        recognition_threshold=0.6,
        recognition_min_margin=0.0,
        pad_min_observations=8,
        max_track_gap_ms=500.0,
        min_iou=0.3,
        min_detection_score=0.6,
        min_face_size_px=80,
        min_blur_variance=40.0,
        min_mean_brightness=40.0,
        max_mean_brightness=215.0,
    )


def _image() -> np.ndarray:
    values = (np.indices((256, 256)).sum(axis=0) % 2 * 200 + 20).astype(np.uint8)
    return np.repeat(values[:, :, None], 3, axis=2)


def _row(sample: str, person: str) -> dict[str, str]:
    return {
        "sample_id": sample,
        "relative_path": f"{person}/{sample}.jpg",
        "verified_person_id": person,
        "source_sha256": sample * 8,
    }


def _meta(row: dict[str, str], score: float, bbox: list[float]) -> dict[str, str]:
    return {
        "relative_path": row["relative_path"],
        "source_sha256": row["source_sha256"],
        "status": "embedded",
        "detection_score": str(score),
        "bbox_xyxy": json.dumps(bbox),
    }


def test_gate_removes_bystander_faces_and_orphaned_known_probes() -> None:
    subject_face = [40.0, 40.0, 200.0, 220.0]
    bystander_face = [230.0, 20.0, 251.0, 49.0]  # ~21 px, like the observed bystanders
    good_a = _row("aaaaaaaa", "0001")
    bystander_a = _row("bbbbbbbb", "0001")
    bystander_b = _row("cccccccc", "0002")
    known_a = _row("dddddddd", "0001")
    known_b = _row("eeeeeeee", "0002")
    impostor = _row("ffffffff", "0009")
    splits = {
        "recognition_enrollment": [good_a, bystander_a, bystander_b],
        "recognition_val_known": [known_a, known_b],
        "recognition_val_impostor": [impostor],
        "recognition_test_known": [],
        "recognition_test_impostor": [],
    }
    metadata = {
        row["relative_path"]: _meta(row, score, box)
        for row, score, box in (
            (good_a, 0.9, subject_face),
            (bystander_a, 0.51, bystander_face),
            (bystander_b, 0.75, bystander_face),
            (known_a, 0.9, subject_face),
            (known_b, 0.9, subject_face),
            (impostor, 0.9, subject_face),
        )
    }

    kept, exclusions = GATE.gate_rows(splits, metadata, lambda _path: _image(), _policy())

    assert kept["recognition_enrollment"] == [good_a]
    assert kept["recognition_val_known"] == [known_a]
    assert kept["recognition_val_impostor"] == [impostor]
    reasons = {str(row["sample_id"]): row["reason"] for row in exclusions}
    assert reasons == {
        "bbbbbbbb": "low_detection_score",
        "cccccccc": "face_too_small",
        "eeeeeeee": GATE.NO_ENROLLMENT_REASON,
    }


def test_gate_fails_closed_on_hash_mismatch() -> None:
    row = _row("aaaaaaaa", "0001")
    meta = _meta(row, 0.9, [40.0, 40.0, 200.0, 220.0])
    meta["source_sha256"] = "0" * 64
    splits = {name: [] for name in GATE.SPLITS}
    splits["recognition_enrollment"] = [row]
    with pytest.raises(GATE.QualityGateError, match="hash mismatch"):
        GATE.gate_rows(splits, {row["relative_path"]: meta}, lambda _path: _image(), _policy())
