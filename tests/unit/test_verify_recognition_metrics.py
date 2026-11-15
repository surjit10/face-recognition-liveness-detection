import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from faceguard.config import sha256_file
from faceguard.recognition_metrics import (
    evaluate_recognition_operating_point,
    select_recognition_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_verify_recognition_metrics",
    PROJECT_ROOT / "scripts" / "verify_recognition_metrics.py",
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)

FIELDS = ("probe_kind", "probe_person_id", "top1_score", "top1_top2_margin", "correct_identity")


def _write_scores(path: Path, known: list[float], impostor: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, score in enumerate(known):
            writer.writerow({"probe_kind": "known", "probe_person_id": f"p{index % 3}",
                             "top1_score": score, "top1_top2_margin": 0.3,
                             "correct_identity": "true"})
        for index, score in enumerate(impostor):
            writer.writerow({"probe_kind": "impostor", "probe_person_id": f"s{index}",
                             "top1_score": score, "top1_top2_margin": 0.1,
                             "correct_identity": "false"})


def _arrays(path: Path) -> dict:
    return VERIFY.load_scores(path)


def _build(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "validation_scores": tmp_path / "val.csv",
        "test_scores": tmp_path / "test.csv",
        "threshold_report": tmp_path / "threshold.json",
        "validation_report": tmp_path / "validation.json",
        "test_report": tmp_path / "test.json",
        "runtime_config": tmp_path / "runtime.yaml",
    }
    _write_scores(paths["validation_scores"], [0.9, 0.8, 0.75, 0.5], [0.4, 0.35, 0.6])
    _write_scores(paths["test_scores"], [0.85, 0.7, 0.55], [0.3, 0.45])
    val = _arrays(paths["validation_scores"])
    selection = select_recognition_threshold(
        val["known_scores"], val["impostor_scores"], policy="recognition_fmr_bound",
        max_fmr=0.0, known_correct=val["known_correct"], known_margins=val["known_margins"],
        impostor_margins=val["impostor_margins"], min_margin=0.0,
    )
    selection["validation_scores_sha256"] = sha256_file(paths["validation_scores"])
    paths["threshold_report"].write_text(json.dumps(selection), encoding="utf-8")
    threshold_hash = sha256_file(paths["threshold_report"])
    for name in ("validation", "test"):
        scores = _arrays(paths[f"{name}_scores"])
        metrics = evaluate_recognition_operating_point(
            scores["known_scores"], scores["impostor_scores"],
            threshold=selection["threshold"], min_margin=0.0,
            known_correct=scores["known_correct"], known_margins=scores["known_margins"],
            impostor_margins=scores["impostor_margins"],
        )
        paths[f"{name}_report"].write_text(json.dumps({
            "scores_sha256": sha256_file(paths[f"{name}_scores"]),
            "threshold_report_sha256": threshold_hash,
            "metrics": metrics,
        }), encoding="utf-8")
    paths["runtime_config"].write_text(
        f"face:\n  threshold: {selection['threshold']!r}\n  min_top1_top2_margin: 0.0\n",
        encoding="utf-8",
    )
    return paths


def test_consistent_evidence_verifies_and_reports_aggregates_only(tmp_path: Path) -> None:
    result = VERIFY.verify(**_build(tmp_path))

    assert result["status"] == "verified"
    assert result["threshold"] == pytest.approx(0.75)
    assert result["validation"]["impostor_false_accepts"] == 0
    assert result["test"]["known_rejections"] == 2  # 0.70 and 0.55 fall below 0.75
    assert result["test"]["impostor_fpir_upper_95"] == pytest.approx(1 - 0.05 ** 0.5)
    rendered = json.dumps(result)
    assert "p0" not in rendered and "s0" not in rendered


def test_edited_published_metric_fails(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    report = json.loads(paths["test_report"].read_text(encoding="utf-8"))
    report["metrics"]["known_rejections"] = 0
    paths["test_report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(VERIFY.VerificationError, match="test.known_rejections"):
        VERIFY.verify(**paths)


def test_changed_scores_break_provenance(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    _write_scores(paths["test_scores"], [0.85, 0.7, 0.56], [0.3, 0.45])

    with pytest.raises(VERIFY.VerificationError, match="not computed from these test scores"):
        VERIFY.verify(**paths)


def test_runtime_threshold_drift_fails(tmp_path: Path) -> None:
    paths = _build(tmp_path)
    paths["runtime_config"].write_text(
        "face:\n  threshold: 0.7\n  min_top1_top2_margin: 0.0\n", encoding="utf-8"
    )

    with pytest.raises(VERIFY.VerificationError, match="face.threshold"):
        VERIFY.verify(**paths)


def test_zero_event_bound_matches_rule_of_three_scale() -> None:
    assert VERIFY.zero_event_upper_bound(274) == pytest.approx(0.01087377784900223)
    assert VERIFY.zero_event_upper_bound(300) == pytest.approx(3 / 300, rel=0.01)


def test_cli_returns_error_when_private_scores_are_absent(tmp_path: Path) -> None:
    code = VERIFY.main(["--validation-scores", str(tmp_path / "missing.csv"),
                        "--output", str(tmp_path / "out.json")])

    assert code == 2
    assert not (tmp_path / "out.json").exists()
