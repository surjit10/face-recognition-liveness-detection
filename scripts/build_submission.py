#!/usr/bin/env python3
"""Build a privacy-safe academic submission ZIP from an explicit allowlist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scan_artifacts import scan  # noqa: E402


PUBLIC_DIRECTORIES = {
    ".github",
    "configs",
    "data",
    "deployment",
    "docs",
    "models",
    "reports",
    "scripts",
    "src",
    "tests",
}
PUBLIC_ROOT_FILES = {
    ".dockerignore",
    ".dvcignore",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "dvc.lock",
    "dvc.yaml",
    "pyproject.toml",
    "requirements-dev.in",
    "requirements-mlops.in",
    "requirements-mlops.lock",
    "requirements-runtime.in",
    "requirements-runtime.lock",
}
LOCAL_ONLY_PREFIXES = (
    "configs/private/",
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "models/private/",
    "models/releases/",
    "reports/private/",
    "reference/archive/",
)
SUBMISSION_EXCLUDED_FILES = {
    "reports/environments/environment_local_before.json",
    "reports/environments/environment_local_runtime.json",
    "reports/phase_receipts/promotion_faceguard-v2-development.json",
}


class SubmissionError(RuntimeError):
    """Raised when a safe submission archive cannot be produced."""


def _git_paths(root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [value.decode("utf-8") for value in result.stdout.split(b"\0") if value]


def select_submission_paths(paths: list[str]) -> list[str]:
    """Return normalized allowlisted paths; private zones remain excluded."""
    selected: list[str] = []
    for raw in sorted(set(paths)):
        relative = raw.replace("\\", "/").lstrip("./")
        if not relative or relative.startswith("../"):
            continue
        if relative in SUBMISSION_EXCLUDED_FILES:
            continue
        parts = Path(relative).parts
        if not parts:
            continue
        if any(relative.startswith(prefix) for prefix in LOCAL_ONLY_PREFIXES):
            if Path(relative).name != ".gitkeep":
                continue
        top = parts[0]
        allowed = relative in PUBLIC_ROOT_FILES or top in PUBLIC_DIRECTORIES
        if allowed:
            selected.append(relative)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(root: Path, output: Path, paths: list[str]) -> dict:
    existing = [relative for relative in paths if (root / relative).is_file()]
    findings = scan(root, existing)
    if findings:
        raise SubmissionError("; ".join(findings))

    entries = [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": _sha256(root / relative),
        }
        for relative in existing
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "public-safe academic submission",
        "file_count": len(entries),
        "files": entries,
        "excluded_zones": [
            ".git and local IDE/tool state",
            ".venv and .venv-mlops",
            "data/raw, data/interim and data/processed",
            "models/private and models/releases",
            "reports/private",
            "mlflow.db and mlruns",
            "dist and reference/archive",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in existing:
            source = root / relative
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
        info = zipfile.ZipInfo("SUBMISSION_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an allowlisted FaceGuard academic submission ZIP."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dist" / "faceguard-academic-submission.zip",
    )
    args = parser.parse_args()
    try:
        selected = select_submission_paths(_git_paths(PROJECT_ROOT))
        manifest = build_archive(PROJECT_ROOT, args.output, selected)
    except (OSError, subprocess.CalledProcessError, SubmissionError) as exc:
        print(f"submission build blocked: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "event": "submission_built",
                "output": str(args.output.resolve()),
                "file_count": manifest["file_count"],
                "classification": manifest["classification"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
