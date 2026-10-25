"""Reusable, hash-verified runtime construction for CLI and desktop entry points."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gallery import Gallery
from .models import InsightFaceEmbedder
from .pad import OnnxPadModel
from .pipeline import VideoVerifier
from .runtime_config import RuntimeConfig, load_runtime_config


@dataclass(frozen=True)
class RuntimeBundle:
    """Fully initialized runtime and its public-safe readiness receipt."""

    config: RuntimeConfig
    verifier: VideoVerifier
    readiness: dict[str, Any]


def build_readiness_report(
    verifier: VideoVerifier,
    config: RuntimeConfig,
) -> dict[str, Any]:
    """Return camera-free, public-safe runtime readiness metadata."""
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": "runtime_ready",
        "release_id": config.release_id,
        "providers": list(config.providers),
        "face_pack": config.insightface_pack,
        "face_runtime_providers": {
            name: list(providers)
            for name, providers in verifier.analyzer.runtime_providers.items()
        },
        "verified_face_model_hashes": verifier.analyzer.model_hashes,
        "pad_model_sha256": verifier.pad_model.model_sha256,
        "gallery_vectors": verifier.gallery.size,
        "gallery_identities": verifier.gallery.identity_count,
        "camera_opened": False,
    }


class RuntimeFactory:
    """Build the configured runtime without downloads or silent fallbacks."""

    @staticmethod
    def load(
        config_path: str | Path,
        *,
        project_root: str | Path,
    ) -> RuntimeBundle:
        config = load_runtime_config(config_path, project_root=project_root)
        analyzer = InsightFaceEmbedder(
            model_root=config.insightface_model_root,
            pack=config.insightface_pack,
            providers=config.providers,
            expected_hashes={
                config.detector_file: config.detector_sha256,
                config.recognizer_file: config.recognizer_sha256,
            },
        )
        pad = OnnxPadModel(
            config.pad_path,
            expected_sha256=config.pad_sha256,
            providers=config.providers,
            preprocess=config.pad_preprocess,
        )
        gallery = Gallery.load(
            config.gallery_path,
            expected_sha256=config.gallery_sha256,
            expected_embedding_model_sha256=config.recognizer_sha256,
        )
        verifier = VideoVerifier(
            analyzer=analyzer,
            pad_model=pad,
            gallery=gallery,
            policy=config.pipeline_policy,
            model_version=config.release_id,
        )
        return RuntimeBundle(
            config=config,
            verifier=verifier,
            readiness=build_readiness_report(verifier, config),
        )
