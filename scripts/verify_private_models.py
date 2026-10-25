#!/usr/bin/env python3
"""Verify private model artifacts, hashes, and inference readiness without leaking biometric data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.gallery import Gallery, GalleryError  # noqa: E402
from faceguard.models import InsightFaceEmbedder, ModelPackError  # noqa: E402
from faceguard.pad import OnnxPadModel, PadInferenceError  # noqa: E402
from faceguard.pipeline import VideoVerifier  # noqa: E402
from faceguard.runtime_config import ConfigurationError, load_runtime_config  # noqa: E402


class PrivateVerificationError(RuntimeError):
    """Raised when private model verification fails."""


def verify_private_artifacts(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
    readiness_report_path: Path | None = None,
) -> dict:
    if not config_path.is_file():
        raise PrivateVerificationError(f"Config file not found: {config_path}")

    config = load_runtime_config(config_path, project_root=project_root)

    # 1. Load and hash check InsightFace pack
    try:
        analyzer = InsightFaceEmbedder(
            model_root=config.insightface_model_root,
            pack=config.insightface_pack,
            providers=config.providers,
            expected_hashes={
                config.detector_file: config.detector_sha256,
                config.recognizer_file: config.recognizer_sha256,
            },
        )
    except ModelPackError as exc:
        raise PrivateVerificationError(f"InsightFace verification failed: {exc}") from exc

    # 2. Load and hash check PAD model
    try:
        pad = OnnxPadModel(
            config.pad_path,
            expected_sha256=config.pad_sha256,
            providers=config.providers,
            preprocess=config.pad_preprocess,
        )
    except PadInferenceError as exc:
        raise PrivateVerificationError(f"PAD model verification failed: {exc}") from exc

    # 3. Load and hash check private Gallery
    try:
        gallery = Gallery.load(
            config.gallery_path,
            expected_sha256=config.gallery_sha256,
            expected_embedding_model_sha256=config.recognizer_sha256,
        )
    except GalleryError as exc:
        raise PrivateVerificationError(f"Gallery verification failed: {exc}") from exc

    # 4. Construct verifier and test input contract with synthetic non-biometric frame
    verifier = VideoVerifier(
        analyzer=analyzer,
        pad_model=pad,
        gallery=gallery,
        policy=config.pipeline_policy,
        model_version=config.release_id,
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = verifier.process(dummy_frame, 100.0)

    # Check event safety: ensure no raw embedding vectors in event dictionary
    event = result.event()
    for key, val in event.items():
        if isinstance(val, (list, np.ndarray)) and len(val) > 10:
            raise PrivateVerificationError(f"Event contains raw vector data in key: {key}")

    readiness = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": "runtime_ready",
        "release_id": config.release_id,
        "providers": list(config.providers),
        "face_pack": config.insightface_pack,
        "face_runtime_providers": {
            name: list(providers)
            for name, providers in analyzer.runtime_providers.items()
        },
        "verified_face_model_hashes": analyzer.model_hashes,
        "pad_model_sha256": pad.model_sha256,
        "gallery_vectors": gallery.size,
        "gallery_identities": gallery.identity_count,
        "camera_opened": False,
        "synthetic_test_status": result.status,
    }

    if readiness_report_path is not None:
        readiness_report_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(readiness, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=readiness_report_path.parent,
            prefix=f".{readiness_report_path.name}.",
            delete=False,
        ) as handle:
            handle.write(rendered)
            temporary = Path(handle.name)
        os.replace(temporary, readiness_report_path)

    return readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify private models without exposing biometric data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.yaml",
        help="Path to runtime configuration YAML",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=PROJECT_ROOT / "reports/environments/runtime_readiness.json",
        help="Output path for readiness JSON receipt",
    )
    args = parser.parse_args()

    try:
        report = verify_private_artifacts(args.config, PROJECT_ROOT, args.readiness_report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (PrivateVerificationError, ConfigurationError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
