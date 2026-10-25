#!/usr/bin/env python3
"""Verify and promote the private PAD bundle into the local runtime layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


EXPECTED_MEMBERS = {
    "mobilenetv3_pad.onnx",
    "mobilenetv3_pad_best.pth",
    "pad_config.json",
    "pad_metrics.json",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.pad_bundle.is_file():
        parser.error(f"PAD bundle does not exist: {args.pad_bundle}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.pad_bundle) as archive:
        members = set(archive.namelist())
        if members != EXPECTED_MEMBERS:
            parser.error(
                f"PAD bundle members mismatch; missing={sorted(EXPECTED_MEMBERS-members)}, "
                f"unknown={sorted(members-EXPECTED_MEMBERS)}"
            )
        corrupt = archive.testzip()
        if corrupt is not None:
            parser.error(f"PAD bundle CRC failure: {corrupt}")
        payload = {name: archive.read(name) for name in EXPECTED_MEMBERS}
    metrics = json.loads(payload["pad_metrics.json"])
    expected_hashes = {
        "mobilenetv3_pad.onnx": metrics["onnx_sha256"],
        "mobilenetv3_pad_best.pth": metrics["best_checkpoint_sha256"],
        "pad_config.json": metrics["config_sha256"],
    }
    for name, expected in expected_hashes.items():
        actual = digest(payload[name])
        if actual != expected:
            parser.error(f"Hash mismatch for {name}: expected {expected}, got {actual}")
    promoted: dict[str, str] = {}
    for name in sorted(EXPECTED_MEMBERS):
        target = args.output_dir / name
        if target.exists() and not args.overwrite:
            if digest(target.read_bytes()) != digest(payload[name]):
                parser.error(f"Refusing to overwrite different artifact: {target}")
            promoted[name] = digest(payload[name])
            continue
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=args.output_dir, prefix=f".{name}.", delete=False
        ) as handle:
            handle.write(payload[name])
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        promoted[name] = digest(payload[name])
    print(json.dumps({"output_dir": str(args.output_dir), "artifacts": promoted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
