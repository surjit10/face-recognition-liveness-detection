#!/usr/bin/env python3
"""Validate official CelebA-Spoof metadata and freeze PAD split manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.pad_manifest import (  # noqa: E402
    PadManifestError,
    prepare_celeba_spoof_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-subject-fraction", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int, default=100_000)
    parser.add_argument("--validation-limit", type=int, default=20_000)
    parser.add_argument("--test-limit", type=int, default=40_000)
    parser.add_argument("--acknowledge-non-commercial-terms", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_non_commercial_terms:
        parser.exit(
            2,
            "blocked: review the official CelebA-Spoof non-commercial research "
            "agreement and pass --acknowledge-non-commercial-terms\n",
        )
    try:
        report = prepare_celeba_spoof_manifests(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            seed=args.seed,
            validation_subject_fraction=args.validation_subject_fraction,
            train_limit=args.train_limit,
            validation_limit=args.validation_limit,
            test_limit=args.test_limit,
        )
    except PadManifestError as exc:
        parser.exit(2, f"PAD manifest preparation blocked: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
