#!/usr/bin/env python3
"""Install a user-supplied official InsightFace pack without network downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REQUIRED = {
    "det_2.5g.onnx": "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-zip", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--pack", default="buffalo_m", choices=("buffalo_m",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--acknowledge-non-commercial-model-terms", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_non_commercial_model_terms:
        parser.exit(
            2,
            "blocked: review the InsightFace supplied-model terms and pass "
            "--acknowledge-non-commercial-model-terms\n",
        )
    if not args.pack_zip.is_file():
        parser.error(f"Pack ZIP does not exist: {args.pack_zip}")
    with zipfile.ZipFile(args.pack_zip) as archive:
        if archive.testzip() is not None:
            parser.error("Pack ZIP failed its CRC check")
        by_name: dict[str, str] = {}
        for member in archive.namelist():
            pure = PurePosixPath(member)
            if pure.name in REQUIRED:
                if pure.name in by_name:
                    parser.error(f"Pack contains duplicate model filename: {pure.name}")
                by_name[pure.name] = member
        missing = set(REQUIRED).difference(by_name)
        if missing:
            parser.error(f"Pack is missing required models: {sorted(missing)}")
        payload = {name: archive.read(by_name[name]) for name in REQUIRED}
    for name, expected in REQUIRED.items():
        actual = sha256(payload[name])
        if actual != expected:
            parser.error(f"Official hash mismatch for {name}: expected {expected}, got {actual}")
    output = args.model_root / "models" / args.pack
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        target = output / name
        if target.exists() and not args.overwrite:
            if sha256(target.read_bytes()) != REQUIRED[name]:
                parser.error(f"Refusing to overwrite different file: {target}")
            continue
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=output, prefix=f".{name}.", delete=False
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    print(
        json.dumps(
            {"pack": args.pack, "output": str(output), "verified_models": REQUIRED},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
