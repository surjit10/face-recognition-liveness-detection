#!/usr/bin/env python3
"""Recompute recognition metrics from private search scores and verify the published reports.

This is the DVC ``evaluate`` stage. It re-derives the validation threshold and the
validation/test operating points from the private per-probe score CSVs, checks them
against the committed reports and ``configs/runtime.yaml``, and writes an aggregate,
identity-free metrics file for ``dvc metrics show`` / ``dvc metrics diff``.

The committed reports carry hand-written disclosures (data-quality gate, superseded
results, limitations), so this stage verifies them instead of overwriting them. Any
drift between scores, threshold, config and reports fails the stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.recognition_metrics import (  # noqa: E402
    RecognitionMetricError,
    evaluate_recognition_operating_point,
    select_recognition_threshold,
)

SCORE_FIELDS = {"probe_kind", "top1_score", "top1_top2_margin", "correct_identity"}
POINT_METRICS = (
    "threshold",
    "min_top1_top2_margin",
    "known_searches",
    "known_correct_accepts",
    "known_misidentification_accepts",
    "known_rejections",
    "known_identification_failure_rate",
    "known_misidentification_accept_rate",
    "impostor_searches",
    "impostor_false_accepts",
    "impostor_false_positive_identification_rate",
)
CURVE_FIELDS = ("threshold", "false_matches", "false_non_matches")
TOLERANCE = 1e-12


class VerificationError(RuntimeError):
    """Published recognition evidence does not match its private inputs."""


def _boolean(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise VerificationError(f"Expected true/false, got {value!r}")
    return lowered == "true"


def load_scores(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or SCORE_FIELDS.difference(rows[0]):
        raise VerificationError(f"{path.name} requires columns {sorted(SCORE_FIELDS)}")
    known = [row for row in rows if row["probe_kind"] == "known"]
    impostor = [row for row in rows if row["probe_kind"] == "impostor"]
    if not known or not impostor or len(known) + len(impostor) != len(rows):
        raise VerificationError(f"{path.name} must hold non-empty known and impostor rows only")
    return {
        "known_scores": np.asarray([float(r["top1_score"]) for r in known]),
        "impostor_scores": np.asarray([float(r["top1_score"]) for r in impostor]),
        "known_correct": np.asarray([_boolean(r["correct_identity"]) for r in known]),
        "known_margins": np.asarray([float(r["top1_top2_margin"]) for r in known]),
        "impostor_margins": np.asarray([float(r["top1_top2_margin"]) for r in impostor]),
    }


def zero_event_upper_bound(searches: int, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound for 0 events in ``searches`` trials."""
    return 1.0 - (1.0 - confidence) ** (1.0 / searches)


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    return a == b


def _compare(label: str, expected: dict, actual: dict, keys, problems: list[str]) -> None:
    for key in keys:
        if key not in expected:
            problems.append(f"{label}: published report lacks {key}")
        elif not _close(expected[key], actual[key]):
            problems.append(f"{label}.{key}: published {expected[key]!r}, recomputed {actual[key]!r}")


def verify(
    *,
    validation_scores: Path,
    test_scores: Path,
    threshold_report: Path,
    validation_report: Path,
    test_report: Path,
    runtime_config: Path,
) -> dict[str, Any]:
    threshold_doc = json.loads(threshold_report.read_text(encoding="utf-8"))
    reports = {
        "validation": json.loads(validation_report.read_text(encoding="utf-8")),
        "test": json.loads(test_report.read_text(encoding="utf-8")),
    }
    face = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))["face"]
    score_paths = {"validation": validation_scores, "test": test_scores}
    hashes = {name: sha256_file(path) for name, path in score_paths.items()}
    threshold_hash = sha256_file(threshold_report)
    problems: list[str] = []

    # 1. Provenance: every report must point at exactly these inputs.
    if threshold_doc.get("validation_scores_sha256") != hashes["validation"]:
        problems.append("threshold report was not computed from these validation scores")
    for name, report in reports.items():
        if report.get("scores_sha256") != hashes[name]:
            problems.append(f"{name} report was not computed from these {name} scores")
        if report.get("threshold_report_sha256") != threshold_hash:
            problems.append(f"{name} report references a different threshold report")

    # 2. Threshold selection is re-run on validation only and must reproduce exactly.
    validation = load_scores(validation_scores)
    selection = select_recognition_threshold(
        validation["known_scores"],
        validation["impostor_scores"],
        policy=threshold_doc["policy"],
        max_fmr=threshold_doc.get("max_fmr"),
        known_correct=validation["known_correct"],
        known_margins=validation["known_margins"],
        impostor_margins=validation["impostor_margins"],
        min_margin=float(threshold_doc["min_top1_top2_margin"]),
    )
    threshold = float(selection["threshold"])
    min_margin = float(selection["min_top1_top2_margin"])
    if not _close(threshold, threshold_doc["threshold"]):
        problems.append(
            f"threshold: published {threshold_doc['threshold']!r}, recomputed {threshold!r}"
        )
    published_curve = threshold_doc.get("curve", [])
    if len(published_curve) != len(selection["curve"]):
        problems.append("threshold curve length differs from recomputation")
    else:
        for index, (old, new) in enumerate(zip(published_curve, selection["curve"])):
            if not all(_close(old[k], new[k]) for k in CURVE_FIELDS):
                problems.append(f"threshold curve differs at row {index}")
                break
    if not _close(float(face["threshold"]), threshold):
        problems.append(f"configs/runtime.yaml face.threshold {face['threshold']!r} != {threshold!r}")
    if not _close(float(face["min_top1_top2_margin"]), min_margin):
        problems.append("configs/runtime.yaml face.min_top1_top2_margin differs from calibration")

    # 3. Frozen operating point on validation and the once-scored test split.
    summary: dict[str, Any] = {}
    for name, report in reports.items():
        scores = validation if name == "validation" else load_scores(test_scores)
        metrics = evaluate_recognition_operating_point(
            scores["known_scores"],
            scores["impostor_scores"],
            threshold=threshold,
            min_margin=min_margin,
            known_correct=scores["known_correct"],
            known_margins=scores["known_margins"],
            impostor_margins=scores["impostor_margins"],
        )
        _compare(name, report.get("metrics", {}), metrics, POINT_METRICS, problems)
        # Threshold and margin are shared, so they are reported once at top level.
        summary[name] = {key: metrics[key] for key in POINT_METRICS[2:]}
        if metrics["impostor_false_accepts"] == 0:
            bound = zero_event_upper_bound(metrics["impostor_searches"])
            summary[name]["impostor_fpir_upper_95"] = bound
            published = report.get("metrics", {}).get("impostor_zero_event_upper_bound_95")
            if published is not None and not _close(published.get("upper_95"), bound):
                problems.append(f"{name}: published zero-event upper bound does not match")

    if problems:
        raise VerificationError("; ".join(problems))
    return {
        "schema_version": 1,
        "status": "verified",
        "threshold": threshold,
        "min_top1_top2_margin": min_margin,
        "threshold_policy": threshold_doc["policy"],
        "validation": summary["validation"],
        "test": summary["test"],
        "inputs_sha256": {
            "validation_scores": hashes["validation"],
            "test_scores": hashes["test"],
            "threshold_report": threshold_hash,
        },
        "privacy": "aggregate counts and rates only; no probe, identity or embedding values",
    }


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    scores_dir = PROJECT_ROOT / "data/interim/recognition_scores"
    evaluation = PROJECT_ROOT / "reports/evaluation"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validation-scores", type=Path,
                        default=scores_dir / "recognition_validation_scores.csv")
    parser.add_argument("--test-scores", type=Path,
                        default=scores_dir / "recognition_test_scores.csv")
    parser.add_argument("--threshold-report", type=Path,
                        default=evaluation / "recognition_threshold.json")
    parser.add_argument("--validation-report", type=Path,
                        default=evaluation / "recognition_validation.json")
    parser.add_argument("--test-report", type=Path, default=evaluation / "recognition_test.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument("--output", type=Path, default=evaluation / "recognition_metrics.json")
    args = parser.parse_args(argv)
    try:
        result = verify(
            validation_scores=args.validation_scores,
            test_scores=args.test_scores,
            threshold_report=args.threshold_report,
            validation_report=args.validation_report,
            test_report=args.test_report,
            runtime_config=args.config,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            RecognitionMetricError, VerificationError) as exc:
        print(f"recognition verification failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    _write_atomic(args.output, rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
