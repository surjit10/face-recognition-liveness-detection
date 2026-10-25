#!/usr/bin/env python3
"""Generate and validate a concrete, schema-valid FaceGuard release manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402


class ReleaseManifestError(RuntimeError):
    """Raised when release manifest generation or validation fails."""


def get_git_info(project_root: Path) -> tuple[str, bool, list[str]]:
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"

    try:
        status_output = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty_files = [line.strip() for line in status_output.splitlines() if line.strip()]
        is_clean = len(dirty_files) == 0
    except Exception:
        is_clean = False
        dirty_files = ["unknown"]

    return commit, is_clean, dirty_files


def build_manifest(config_path: Path, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if not config_path.is_file():
        raise ReleaseManifestError(f"Runtime configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ReleaseManifestError("Invalid runtime config schema")

    face_cfg = config.get("face", {})
    pad_cfg = config.get("pad", {})
    runtime_cfg = config.get("runtime", {})

    # Verify and hash model artifacts
    model_root = project_root / face_cfg.get("model_root", "")
    pack = face_cfg.get("pack", "buffalo_m")
    pack_dir = model_root / "models" / pack if (model_root / "models" / pack).is_dir() else model_root
    detector_path = pack_dir / face_cfg.get("detector_file", "")
    recognizer_path = pack_dir / face_cfg.get("recognizer_file", "")
    pad_path = project_root / pad_cfg.get("file", "")
    gallery_path = project_root / face_cfg.get("gallery_file", "")

    artifacts = {}
    for name, path, expected_hash in [
        ("detector", detector_path, face_cfg.get("detector_sha256")),
        ("recognizer", recognizer_path, face_cfg.get("recognizer_sha256")),
        ("pad_onnx", pad_path, pad_cfg.get("sha256")),
        ("gallery", gallery_path, face_cfg.get("gallery_sha256")),
    ]:
        if not path.is_file():
            raise ReleaseManifestError(f"Required artifact {name} missing at: {path}")
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise ReleaseManifestError(
                f"Artifact {name} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        artifacts[name] = {
            "relative_path": str(path.relative_to(project_root)),
            "sha256": actual_hash,
        }

    # Dependency lock
    lock_file = project_root / "requirements-runtime.lock"
    lock_hash = sha256_file(lock_file) if lock_file.is_file() else "missing"

    # Git metadata
    git_commit, git_clean, dirty_files = get_git_info(project_root)

    # Optional report references
    reports = {}
    for report_key, report_rel in [
        ("readiness", "reports/environments/runtime_readiness.json"),
        ("recognition_validation", "reports/evaluation/recognition_validation.json"),
        ("recognition_test", "reports/evaluation/recognition_test.json"),
        ("recognition_threshold", "reports/evaluation/recognition_threshold.json"),
        ("prototype_camera_benchmark", "reports/benchmarks/prototype_camera.json"),
    ]:
        report_p = project_root / report_rel
        if report_p.is_file():
            reports[report_key] = {
                "relative_path": report_rel,
                "sha256": sha256_file(report_p),
            }

    manifest = {
        "schema_version": 1,
        "release_id": config.get("release_id", "unspecified"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_commit,
            "clean": git_clean,
            "uncommitted_files": dirty_files,
        },
        "environment": {
            "python": sys.version.split()[0],
            "runtime_lock_sha256": lock_hash,
        },
        "artifacts": artifacts,
        "configuration": {
            "providers": config.get("providers", ["CPUExecutionProvider"]),
            "face": {
                "threshold": float(face_cfg.get("threshold", 0.0)),
                "min_top1_top2_margin": float(face_cfg.get("min_top1_top2_margin", 0.0)),
            },
            "pad": {
                "threshold": float(pad_cfg.get("threshold", 0.0)),
                "crop_scale": float(pad_cfg.get("crop_scale", 1.3)),
                "max_clipped_fraction": float(pad_cfg.get("max_clipped_fraction", 0.2)),
                "input_size": int(pad_cfg.get("input_size", 224)),
                "mean": list(pad_cfg.get("mean", [0.485, 0.456, 0.406])),
                "std": list(pad_cfg.get("std", [0.229, 0.224, 0.225])),
            },
            "runtime": runtime_cfg,
        },
        "reports": reports,
    }
    validate_manifest(manifest, project_root)
    return manifest


def validate_manifest(manifest: dict[str, Any], project_root: Path = PROJECT_ROOT) -> None:
    if manifest.get("schema_version") != 1:
        raise ReleaseManifestError("Release manifest schema_version must be 1")
    if not isinstance(manifest.get("release_id"), str) or not manifest["release_id"].strip():
        raise ReleaseManifestError("Release manifest requires a valid non-empty release_id")

    artifacts = manifest.get("artifacts", {})
    required_artifacts = {"detector", "recognizer", "pad_onnx", "gallery"}
    if not required_artifacts.issubset(artifacts):
        raise ReleaseManifestError(f"Missing artifacts in manifest: {required_artifacts - set(artifacts)}")

    for name, item in artifacts.items():
        if not isinstance(item, dict) or "sha256" not in item or "relative_path" not in item:
            raise ReleaseManifestError(f"Malformed artifact entry: {name}")
        art_path = project_root / item["relative_path"]
        if not art_path.is_file():
            raise ReleaseManifestError(f"Artifact {name} not found at {art_path}")
        actual_sha = sha256_file(art_path)
        if actual_sha != item["sha256"]:
            raise ReleaseManifestError(
                f"Artifact {name} hash verification failed: {actual_sha} != {item['sha256']}"
            )


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate FaceGuard release manifest.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.yaml",
        help="Path to runtime configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/release_manifest.json",
        help="Path to output manifest JSON file",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Block manifest generation if git working tree has uncommitted changes",
    )
    args = parser.parse_args()

    try:
        manifest = build_manifest(args.config, PROJECT_ROOT)
        if args.require_clean_git and not manifest["git"]["clean"]:
            raise ReleaseManifestError(
                f"Working tree is not clean: {manifest['git']['uncommitted_files']}"
            )
        write_manifest(manifest, args.output)
        print(
            json.dumps(
                {
                    "event": "release_manifest_generated",
                    "output": str(args.output),
                    "release_id": manifest["release_id"],
                    "git_commit": manifest["git"]["commit"],
                    "git_clean": manifest["git"]["clean"],
                },
                indent=2,
            )
        )
        return 0
    except (ReleaseManifestError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
