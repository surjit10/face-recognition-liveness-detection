"""Private embedding extraction from a frozen or audit-stage image manifest."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import (
    index_decisions,
    load_yaml_mapping,
    require_decision,
    require_processing_permission,
    resolve_under,
    sha256_file,
)
from .models import InsightFaceEmbedder


class ExtractionError(RuntimeError):
    """Embedding extraction cannot safely continue."""


METADATA_COLUMNS = (
    "candidate_person_id",
    "relative_path",
    "source_sha256",
    "status",
    "face_count",
    "detection_score",
    "bbox_xyxy",
    "embedding_row",
    "model_pack",
    "error",
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"candidate_person_id", "relative_path", "sha256", "decode_ok", "include", "split"}
    missing = required.difference(rows[0].keys() if rows else set())
    if not rows:
        raise ExtractionError(f"Manifest is empty: {path}")
    if missing:
        raise ExtractionError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def _selected(row: dict[str, str], mode: str) -> bool:
    if row["decode_ok"].strip().lower() != "true":
        return False
    include = row["include"].strip().lower()
    if include == "false":
        return False
    if mode == "audit":
        return include in {"true", "unresolved"}
    if mode == "enrollment":
        return include == "true" and row["split"].strip() == "enrollment"
    raise ExtractionError(f"Unsupported extraction mode: {mode}")


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def extract_embeddings(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    decisions_path: str | Path,
    model_root: str | Path,
    output_dir: str | Path,
    processing_context: str,
    mode: str,
    providers: tuple[str, ...],
    pack: str = "buffalo_m",
    overwrite: bool = False,
) -> dict[str, Any]:
    decisions = load_yaml_mapping(decisions_path)
    require_processing_permission(decisions, context=processing_context)
    indexed_decisions = index_decisions(decisions)
    model_decision = require_decision(indexed_decisions, "INSIGHTFACE_PACK")
    model_value = model_decision.get("value")
    if not isinstance(model_value, dict) or model_value.get("selected_pack") != pack:
        raise ExtractionError(f"INSIGHTFACE_PACK does not freeze selected_pack={pack}")
    detector_file = model_value.get("detector_file")
    detector_sha256 = model_value.get("detector_sha256")
    recognizer_file = model_value.get("recognizer_file")
    recognizer_sha256 = model_value.get("recognizer_sha256")
    frozen_fields = (detector_file, detector_sha256, recognizer_file, recognizer_sha256)
    if not all(isinstance(value, str) and value for value in frozen_fields):
        raise ExtractionError("INSIGHTFACE_PACK file names and SHA-256 values are incomplete")
    expected_hashes = {
        detector_file: detector_sha256,
        recognizer_file: recognizer_sha256,
    }
    if mode == "enrollment":
        for decision_id in ("FRAD_LABEL_MEANING", "FRAD_COHORT", "FRAD_SPLIT"):
            require_decision(indexed_decisions, decision_id)
    manifest = Path(manifest_path)
    rows = _read_manifest(manifest)
    root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    embeddings_path = output / "frad_embeddings.npz"
    metadata_path = output / "frad_embedding_metadata.csv"
    report_path = output / "embedding_extraction.json"
    if not overwrite:
        existing = [path for path in (embeddings_path, metadata_path, report_path) if path.exists()]
        if existing:
            raise ExtractionError(f"Outputs exist; pass --overwrite: {existing}")

    embedder = InsightFaceEmbedder(
        model_root=model_root,
        pack=pack,
        providers=providers,
        expected_hashes=expected_hashes,
    )
    vectors: list[np.ndarray] = []
    person_ids: list[str] = []
    relative_paths: list[str] = []
    source_hashes: list[str] = []
    metadata: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()

    for row in rows:
        if not _selected(row, mode):
            continue
        relative_path = row["relative_path"]
        try:
            image_path = resolve_under(root, relative_path)
            if sha256_file(image_path) != row["sha256"]:
                raise ExtractionError("source_hash_mismatch")
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image_rgb = np.asarray(image, dtype=np.uint8)
            observation = embedder.embed_rgb(image_rgb)
            status = "embedded" if observation.error is None else "rejected"
            error = observation.error or ""
        except Exception as exc:  # per-file quarantine; type is recorded, not biometric content
            observation = None
            status = "rejected"
            error = type(exc).__name__

        embedding_row: int | str = ""
        if observation is not None and observation.embedding is not None:
            embedding_row = len(vectors)
            vectors.append(observation.embedding)
            person_ids.append(row["candidate_person_id"])
            relative_paths.append(relative_path)
            source_hashes.append(row["sha256"])
        outcome_counts[error or "embedded"] += 1
        metadata.append(
            {
                "candidate_person_id": row["candidate_person_id"],
                "relative_path": relative_path,
                "source_sha256": row["sha256"],
                "status": status,
                "face_count": "" if observation is None else observation.face_count,
                "detection_score": ""
                if observation is None or observation.detection_score is None
                else observation.detection_score,
                "bbox_xyxy": ""
                if observation is None or observation.bbox_xyxy is None
                else json.dumps(observation.bbox_xyxy),
                "embedding_row": embedding_row,
                "model_pack": pack,
                "error": error,
            }
        )

    if not vectors:
        raise ExtractionError("No valid embeddings were produced; no output was promoted")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(matrix).all() or not np.allclose(norms, 1.0, atol=1e-5):
        raise ExtractionError("Embedding matrix failed finite/L2-normalization checks")

    model_hashes = embedder.model_hashes
    _atomic_npz(
        embeddings_path,
        embeddings=matrix,
        person_ids=np.asarray(person_ids, dtype=np.str_),
        relative_paths=np.asarray(relative_paths, dtype=np.str_),
        source_sha256=np.asarray(source_hashes, dtype=np.str_),
        model_pack=np.asarray([pack], dtype=np.str_),
        model_hashes_json=np.asarray([json.dumps(model_hashes, sort_keys=True)], dtype=np.str_),
    )
    _atomic_csv(metadata_path, metadata)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processing_context": processing_context,
        "mode": mode,
        "model_pack": pack,
        "model_hashes": model_hashes,
        "providers": list(providers),
        "runtime_providers_by_task": {
            task: list(task_providers)
            for task, task_providers in embedder.runtime_providers.items()
        },
        "manifest_sha256": sha256_file(manifest),
        "selected_samples": len(metadata),
        "embedded_samples": len(vectors),
        "outcomes": dict(sorted(outcome_counts.items())),
        "embeddings_sha256": sha256_file(embeddings_path),
        "metadata_sha256": sha256_file(metadata_path),
        "privacy": "restricted biometric; do not publish",
    }
    _atomic_json(report_path, report)
    return report
