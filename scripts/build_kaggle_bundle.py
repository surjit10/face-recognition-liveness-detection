#!/usr/bin/env python3
"""Build a code-only Kaggle source bundle from an explicit allowlist."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_FILES = (
    "pyproject.toml",
    "requirements-kaggle.in",
    "configs/decisions.yaml",
    "notebooks/02_kaggle_frad_embeddings.ipynb",
    "notebooks/03_kaggle_pad_training.ipynb",
)
ALLOWED_TREES = ("src", "scripts")
FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".mp4",
    ".avi",
    ".onnx",
    ".pt",
    ".pth",
    ".h5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
}


def _selected_files() -> list[Path]:
    selected = [PROJECT_ROOT / relative for relative in ALLOWED_FILES]
    for tree in ALLOWED_TREES:
        selected.extend(
            path
            for path in (PROJECT_ROOT / tree).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Allowlisted files are missing: {missing}")
    forbidden = [path for path in selected if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise RuntimeError(f"Forbidden binary/biometric files selected: {forbidden}")
    return sorted(set(selected))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output
    if output.exists() and not args.overwrite:
        parser.exit(2, f"output exists; pass --overwrite: {output}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _selected_files()
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    import hashlib

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "sha256": digest,
                "files": len(files),
                "privacy": "code/config only; no data, model, report, or reference trees",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
