#!/usr/bin/env python3
"""Hash a local InsightFace pack without loading sessions or downloading files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--pack", default="buffalo_m")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pack_dir = args.model_root.resolve() / "models" / args.pack
    if not pack_dir.is_dir():
        parser.exit(2, f"model pack does not exist: {pack_dir}\n")
    files = sorted(pack_dir.glob("*.onnx"))
    if not files:
        parser.exit(2, f"no ONNX model files found: {pack_dir}\n")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pack": args.pack,
        "files": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files
        },
        "warning": (
            "Hashes prove file identity, not source authenticity or usage rights. "
            "Verify the official source and model terms separately."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
