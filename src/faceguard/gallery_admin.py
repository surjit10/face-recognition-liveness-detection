"""Shared persistent-gallery administration for CLI and desktop workflows."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import sha256_file
from .gallery import Gallery, GalleryError


_MUTATION_LOCK = threading.RLock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,63}$")


def get_manifest_sha256(gallery_path: Path) -> str:
    with np.load(gallery_path, allow_pickle=False) as archive:
        if "manifest_sha256" in archive.files:
            return str(archive["manifest_sha256"].reshape(-1)[0])
    return "0" * 64


def load_gallery_unverified(gallery_path: Path) -> tuple[Gallery, str]:
    """Load an operator-selected gallery while still validating its internal contract."""
    if not gallery_path.is_file():
        raise GalleryError(f"Gallery file not found: {gallery_path}")
    with np.load(gallery_path, allow_pickle=False) as archive:
        model_hash = str(archive["embedding_model_sha256"].reshape(-1)[0])
    manifest_hash = get_manifest_sha256(gallery_path)
    gallery = Gallery.load(
        gallery_path,
        expected_sha256=sha256_file(gallery_path),
        expected_embedding_model_sha256=model_hash,
    )
    return gallery, manifest_hash


def _metadata_path(gallery_path: Path, metadata_path: Path | None) -> Path:
    return metadata_path or gallery_path.with_name("enrollments.json")


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 1, "identities": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GalleryError(f"Unable to read enrollment metadata: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("identities"), dict)
    ):
        raise GalleryError("Enrollment metadata must use schema version 1")
    return document


def list_identities(gallery_path: Path, *, metadata_path: Path | None = None) -> dict[str, Any]:
    gallery, _ = load_gallery_unverified(gallery_path)
    metadata = _load_metadata(_metadata_path(gallery_path, metadata_path))["identities"]
    return {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "identities_count": gallery.identity_count,
        "vectors_count": gallery.size,
        "identities": [
            {
                "person_id": person_id,
                "vectors": gallery.vector_count(person_id),
                "enrolled_at": metadata.get(person_id, {}).get("enrolled_at"),
                "consent_recorded": metadata.get(person_id, {}).get("consent_recorded"),
            }
            for person_id in gallery.identity_ids
        ],
    }


def inspect_identity(
    gallery_path: Path,
    person_id: str,
    *,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    gallery, manifest_hash = load_gallery_unverified(gallery_path)
    if person_id not in gallery.identity_ids:
        raise GalleryError(f"Person ID not found in gallery: {person_id}")
    metadata = _load_metadata(_metadata_path(gallery_path, metadata_path))["identities"].get(
        person_id, {}
    )
    return {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "person_id": person_id,
        "vectors_count": gallery.vector_count(person_id),
        "source_hashes_count": gallery.source_hash_count(person_id),
        "manifest_sha256": manifest_hash,
        "enrolled_at": metadata.get("enrolled_at"),
        "consent_recorded": metadata.get("consent_recorded"),
        "capture_method": metadata.get("capture_method", "legacy_import"),
        "raw_images_retained": metadata.get("raw_images_retained", False),
        "duplicate_candidates_reviewed": metadata.get("duplicate_candidates_reviewed"),
    }


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _update_config_hash(
    config_path: Path,
    *,
    previous_hash: str,
    new_hash: str,
) -> None:
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r"(?m)^(\s*gallery_sha256:\s*)" + re.escape(previous_hash) + r"\s*$")
    updated, count = pattern.subn(r"\g<1>" + new_hash, text)
    if count != 1:
        raise GalleryError("Runtime configuration does not pin the gallery being updated")
    parsed = yaml.safe_load(updated)
    if parsed.get("face", {}).get("gallery_sha256") != new_hash:
        raise GalleryError("Updated runtime configuration did not preserve the gallery hash")
    _atomic_write(config_path, updated.encode("utf-8"), mode=0o644)


def _sidecar_document(gallery_path: Path, gallery: Gallery, new_hash: str) -> dict[str, Any]:
    sidecar_path = gallery_path.with_suffix(".json")
    if sidecar_path.is_file():
        try:
            document = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GalleryError(f"Unable to read gallery sidecar: {exc}") from exc
    else:
        document = {"schema_version": 1, "privacy": "restricted biometric; do not publish"}
    document.update(
        {
            "gallery_file": str(gallery_path),
            "gallery_sha256": new_hash,
            "embedding_model_sha256": gallery.embedding_model_sha256,
            "identities": gallery.identity_count,
            "vectors": gallery.size,
        }
    )
    return document


def _persist_mutation(
    gallery_path: Path,
    gallery: Gallery,
    *,
    manifest_hash: str,
    metadata: dict[str, Any],
    metadata_path: Path,
    runtime_config_path: Path | None,
    previous_hash: str,
) -> str:
    paths = [gallery_path, gallery_path.with_suffix(".json"), metadata_path]
    if runtime_config_path is not None:
        paths.append(runtime_config_path)
    originals = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        new_hash = gallery.save(gallery_path, manifest_sha256=manifest_hash)
        sidecar = _sidecar_document(gallery_path, gallery, new_hash)
        _atomic_write(
            gallery_path.with_suffix(".json"),
            (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        _atomic_write(
            metadata_path,
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        if runtime_config_path is not None:
            _update_config_hash(runtime_config_path, previous_hash=previous_hash, new_hash=new_hash)
        Gallery.load(
            gallery_path,
            expected_sha256=new_hash,
            expected_embedding_model_sha256=gallery.embedding_model_sha256,
        )
        return new_hash
    except Exception as exc:
        for path, payload in originals.items():
            if payload is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write(path, payload, mode=0o644 if path == runtime_config_path else 0o600)
        if isinstance(exc, GalleryError):
            raise
        raise GalleryError(f"Gallery update failed and was rolled back: {exc}") from exc


def duplicate_candidates(
    gallery: Gallery,
    embeddings: np.ndarray,
    *,
    threshold: float,
    min_margin: float,
) -> tuple[str, ...]:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    candidates: set[str] = set()
    for vector in matrix:
        match = gallery.search(vector, threshold=threshold, min_margin=min_margin)
        if match.status == "verified" and match.person_id:
            candidates.add(match.person_id)
    return tuple(sorted(candidates))


def enroll_identity(
    gallery_path: Path,
    person_id: str,
    embeddings: np.ndarray,
    *,
    consent_recorded: bool,
    duplicate_candidates_reviewed: bool,
    duplicate_identity_ids: tuple[str, ...] = (),
    runtime_config_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    label = person_id.strip()
    if not _SAFE_ID.fullmatch(label):
        raise GalleryError(
            "Identity label must start with a letter or number and use at most 64 "
            "letters, numbers, spaces, dots, underscores or hyphens"
        )
    if not consent_recorded:
        raise GalleryError("Enrollment requires recorded consent")
    if duplicate_identity_ids and not duplicate_candidates_reviewed:
        raise GalleryError("Possible duplicate identities require explicit operator review")
    with _MUTATION_LOCK:
        gallery, manifest_hash = load_gallery_unverified(gallery_path)
        if label in gallery.identity_ids:
            raise GalleryError(f"Identity label is already enrolled: {label}")
        previous_hash = sha256_file(gallery_path)
        gallery.enroll(label, embeddings)
        resolved_metadata_path = _metadata_path(gallery_path, metadata_path)
        metadata = _load_metadata(resolved_metadata_path)
        metadata["identities"][label] = {
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "consent_recorded": True,
            "capture_method": "authorized_webcam",
            "vectors": gallery.vector_count(label),
            "raw_images_retained": False,
            "duplicate_candidates": list(duplicate_identity_ids),
            "duplicate_candidates_reviewed": bool(duplicate_candidates_reviewed),
        }
        new_hash = _persist_mutation(
            gallery_path,
            gallery,
            manifest_hash=manifest_hash,
            metadata=metadata,
            metadata_path=resolved_metadata_path,
            runtime_config_path=runtime_config_path,
            previous_hash=previous_hash,
        )
        return {
            "schema_version": 1,
            "status": "enrolled",
            "person_id": label,
            "added_vectors": gallery.vector_count(label),
            "identities": gallery.identity_count,
            "vectors": gallery.size,
            "new_gallery_sha256": new_hash,
            "raw_images_retained": False,
        }


def delete_identity(
    gallery_path: Path,
    person_id: str,
    *,
    dry_run: bool = False,
    confirmed: bool = False,
    runtime_config_path: Path | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    with _MUTATION_LOCK:
        gallery, manifest_hash = load_gallery_unverified(gallery_path)
        if person_id not in gallery.identity_ids:
            raise GalleryError(f"Person ID not found in gallery: {person_id}")
        vectors_to_remove = gallery.vector_count(person_id)
        if gallery.identity_count <= 2:
            raise GalleryError("Deletion blocked: gallery must retain at least two identities.")
        if dry_run:
            return {
                "schema_version": 1,
                "status": "dry_run",
                "person_id": person_id,
                "vectors_to_remove": vectors_to_remove,
                "projected_identities": gallery.identity_count - 1,
                "projected_vectors": gallery.size - vectors_to_remove,
            }
        if not confirmed:
            raise GalleryError(
                f"Deletion requires explicit confirmation (--yes) for person ID: {person_id}"
            )
        previous_hash = sha256_file(gallery_path)
        gallery.remove(person_id)
        resolved_metadata_path = _metadata_path(gallery_path, metadata_path)
        metadata = _load_metadata(resolved_metadata_path)
        metadata["identities"].pop(person_id, None)
        new_hash = _persist_mutation(
            gallery_path,
            gallery,
            manifest_hash=manifest_hash,
            metadata=metadata,
            metadata_path=resolved_metadata_path,
            runtime_config_path=runtime_config_path,
            previous_hash=previous_hash,
        )
        return {
            "schema_version": 1,
            "status": "deleted",
            "person_id": person_id,
            "removed_vectors": vectors_to_remove,
            "remaining_identities": gallery.identity_count,
            "remaining_vectors": gallery.size,
            "new_gallery_sha256": new_hash,
            "note": "Reload the active runtime before further verification.",
        }


def verify_gallery(gallery_path: Path) -> dict[str, Any]:
    gallery, manifest_hash = load_gallery_unverified(gallery_path)
    sidecar_path = gallery_path.with_suffix(".json")
    sidecar_matches = None
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_matches = (
            sidecar.get("gallery_sha256") == sha256_file(gallery_path)
            and sidecar.get("identities") == gallery.identity_count
            and sidecar.get("vectors") == gallery.size
        )
    return {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "identities": gallery.identity_count,
        "vectors": gallery.size,
        "gallery_sha256": sha256_file(gallery_path),
        "manifest_sha256": manifest_hash,
        "sidecar_consistent": sidecar_matches,
        "valid": True,
    }
