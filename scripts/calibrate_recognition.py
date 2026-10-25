#!/usr/bin/env python3
"""Calibrate recognition threshold from private frozen validation-search scores."""

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

from faceguard.recognition_metrics import (  # noqa: E402
    RecognitionMetricError,
    select_recognition_threshold,
)


def _boolean(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered not in {"true", "false"}:
        raise ValueError(f"Expected true/false, got {value!r}")
    return lowered == "true"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("min_recognition_balanced_error", "recognition_fmr_bound"),
        required=True,
    )
    parser.add_argument("--max-fmr", type=float)
    parser.add_argument("--min-margin", type=float, required=True)
    args = parser.parse_args()
    try:
        with args.validation_scores.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"probe_kind", "top1_score", "top1_top2_margin", "correct_identity"}
        if not rows or required.difference(rows[0]):
            raise ValueError(f"Score CSV requires columns: {sorted(required)}")
        known = [row for row in rows if row["probe_kind"] == "known"]
        impostor = [row for row in rows if row["probe_kind"] == "impostor"]
        if len(known) + len(impostor) != len(rows) or not known or not impostor:
            raise ValueError("Score CSV must contain non-empty known and impostor rows only")
        report = select_recognition_threshold(
            np.asarray([float(row["top1_score"]) for row in known]),
            np.asarray([float(row["top1_score"]) for row in impostor]),
            policy=args.policy,
            max_fmr=args.max_fmr,
            known_correct=np.asarray([_boolean(row["correct_identity"]) for row in known]),
            known_margins=np.asarray([float(row["top1_top2_margin"]) for row in known]),
            impostor_margins=np.asarray(
                [float(row["top1_top2_margin"]) for row in impostor]
            ),
            min_margin=args.min_margin,
        )
    except (OSError, ValueError, KeyError, RecognitionMetricError) as exc:
        parser.exit(2, f"recognition calibration blocked: {exc}\n")
    report.update(
        {
            "schema_version": 1,
            "score_unit": "one open-set gallery search per validation probe",
            "source_file": args.validation_scores.name,
        }
    )
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
