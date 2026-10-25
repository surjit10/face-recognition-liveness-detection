#!/usr/bin/env python3
"""Validate frozen recognition manifests and their private source files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.recognition_manifest import (  # noqa: E402
    RecognitionManifestError,
    validate_recognition_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--enrollment", type=Path, required=True)
    parser.add_argument("--val-known", type=Path, required=True)
    parser.add_argument("--val-impostor", type=Path, required=True)
    parser.add_argument("--test-known", type=Path, required=True)
    parser.add_argument("--test-impostor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unknown-session-metadata", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_recognition_manifests(
            dataset_root=args.dataset_root,
            manifests={
                "enrollment": args.enrollment,
                "val_known": args.val_known,
                "val_impostor": args.val_impostor,
                "test_known": args.test_known,
                "test_impostor": args.test_impostor,
            },
            allow_unknown_session_metadata=args.allow_unknown_session_metadata,
        )
    except RecognitionManifestError as exc:
        parser.exit(2, f"recognition manifest validation blocked: {exc}\n")
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
