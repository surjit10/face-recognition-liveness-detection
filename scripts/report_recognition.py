#!/usr/bin/env python3
"""Evaluate frozen recognition settings on validation or untouched test scores."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.recognition_metrics import (  # noqa: E402
    RecognitionMetricError,
    evaluate_recognition_operating_point,
)


def _boolean(value: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"Invalid Boolean value: {value!r}")
    return value == "true"


def _cluster_interval(
    rows: list[dict[str, str]], failure, *, seed: int, resamples: int
) -> dict[str, float]:
    by_person: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_person.setdefault(row["probe_person_id"], []).append(row)
    people = sorted(by_person)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.choice(people, size=len(people), replace=True)
        sample = [row for person in selected for row in by_person[str(person)]]
        estimates[index] = sum(failure(row) for row in sample) / len(sample)
    return {
        "lower_95": float(np.percentile(estimates, 2.5)),
        "upper_95": float(np.percentile(estimates, 97.5)),
        "seed": seed,
        "resamples": resamples,
        "cluster_unit": "pseudonymous person identity",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold-report", type=Path, required=True)
    parser.add_argument("--protocol", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    try:
        with args.scores.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {
            "probe_kind",
            "probe_person_id",
            "top1_score",
            "top1_top2_margin",
            "correct_identity",
        }
        if not rows or required.difference(rows[0]):
            raise ValueError(f"Score CSV requires fields: {sorted(required)}")
        known = [row for row in rows if row["probe_kind"] == "known"]
        impostor = [row for row in rows if row["probe_kind"] == "impostor"]
        threshold_report = json.loads(args.threshold_report.read_text(encoding="utf-8"))
        threshold = float(threshold_report["threshold"])
        min_margin = float(threshold_report["min_top1_top2_margin"])
        metrics = evaluate_recognition_operating_point(
            np.asarray([float(row["top1_score"]) for row in known]),
            np.asarray([float(row["top1_score"]) for row in impostor]),
            threshold=threshold,
            min_margin=min_margin,
            known_correct=np.asarray([_boolean(row["correct_identity"]) for row in known]),
            known_margins=np.asarray([float(row["top1_top2_margin"]) for row in known]),
            impostor_margins=np.asarray(
                [float(row["top1_top2_margin"]) for row in impostor]
            ),
        )
        def accepted(row: dict[str, str]) -> bool:
            return (
                float(row["top1_score"]) >= threshold
                and float(row["top1_top2_margin"]) >= min_margin
            )

        metrics["confidence_intervals"] = {
            "known_identification_failure_rate": _cluster_interval(
                known,
                lambda row: not (accepted(row) and _boolean(row["correct_identity"])),
                seed=args.bootstrap_seed,
                resamples=args.bootstrap_resamples,
            ),
            "impostor_false_positive_identification_rate": _cluster_interval(
                impostor,
                accepted,
                seed=args.bootstrap_seed,
                resamples=args.bootstrap_resamples,
            ),
        }
        report = {
            "schema_version": 1,
            "protocol": args.protocol,
            "official_benchmark": False,
            "score_unit": "one gallery search per probe image",
            "scores_sha256": sha256_file(args.scores),
            "threshold_report_sha256": sha256_file(args.threshold_report),
            "metrics": metrics,
            "limitations": [
                "Exploratory unknown-session split; not an official benchmark.",
                "Confidence intervals resample person identities, but sessions/devices are unknown.",
            ],
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RecognitionMetricError) as exc:
        parser.exit(2, f"recognition report blocked: {exc}\n")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
