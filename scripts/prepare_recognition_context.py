#!/usr/bin/env python3
"""Create a non-destructive, context-preserving recognition dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.preprocessing import (  # noqa: E402
    PreprocessingError,
    prepare_recognition_context_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--detector-max-side", type=int, default=960)
    parser.add_argument("--crop-factor", type=float, default=2.0)
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--fallback-max-side", type=int, default=1600)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--multiple-face-policy",
        choices=("full_frame", "largest"),
        default="full_frame",
        help="Keep the full frame or crop the largest detected face for multi-face inputs.",
    )
    args = parser.parse_args()
    try:
        result = prepare_recognition_context_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            manifest_path=args.manifest,
            report_path=args.report,
            detector_model=args.detector_model,
            workers=args.workers,
            detector_max_side=args.detector_max_side,
            crop_factor=args.crop_factor,
            crop_size=args.crop_size,
            fallback_max_side=args.fallback_max_side,
            score_threshold=args.score_threshold,
            jpeg_quality=args.jpeg_quality,
            multiple_face_policy=args.multiple_face_policy,
        )
    except PreprocessingError as exc:
        parser.exit(2, f"preprocessing blocked: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
