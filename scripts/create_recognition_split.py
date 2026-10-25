#!/usr/bin/env python3
"""Create the frozen exploratory open-set recognition manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.recognition_split import (  # noqa: E402
    RecognitionSplitError,
    create_exploratory_recognition_split,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unknown-identity-fraction", type=float, default=0.2)
    parser.add_argument("--known-probe-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = create_exploratory_recognition_split(
            embedding_archive=args.embedding_archive,
            output_dir=args.output_dir,
            seed=args.seed,
            unknown_identity_fraction=args.unknown_identity_fraction,
            known_probe_fraction=args.known_probe_fraction,
            overwrite=args.overwrite,
        )
    except RecognitionSplitError as exc:
        parser.exit(2, f"recognition split blocked: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
