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
from .models import EmbeddingObservation, InsightFaceEmbedder


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
    "selected_input_variant",
    "padding_ltrb",
    "detection_attempts_json",
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


def _normalized_padding_ratios(values: tuple[float, ...]) -> tuple[float, ...]:
    ratios: list[float] = []
    for value in values:
        ratio = float(value)
        if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
            raise ExtractionError(
                "Retry padding ratios must be finite values in the interval (0, 1]"
            )
        if ratio not in ratios:
            ratios.append(ratio)
    return tuple(ratios)


def _padded_canvas(
    image_rgb: np.ndarray, ratio: float
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center an image on a neutral canvas without modifying source pixels."""
    height, width = image_rgb.shape[:2]
    pad_x = max(1, int(round(width * ratio)))
    pad_y = max(1, int(round(height * ratio)))
    channel_median = np.median(image_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    canvas = np.empty((height + 2 * pad_y, width + 2 * pad_x, 3), dtype=np.uint8)
    canvas[...] = channel_median
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = image_rgb
    return np.ascontiguousarray(canvas), (pad_x, pad_y, pad_x, pad_y)


def _attempt_record(
    variant: str, observation: EmbeddingObservation
) -> dict[str, Any]:
    return {
        "variant": variant,
        "face_count": observation.face_count,
        "detection_score": observation.detection_score,
        "bbox_xyxy": observation.bbox_xyxy,
        "error": observation.error,
    }


def _embed_with_padding_retries(
    embedder: InsightFaceEmbedder,
    image_rgb: np.ndarray,
    padding_ratios: tuple[float, ...],
) -> tuple[
    EmbeddingObservation,
    str,
    tuple[int, int, int, int],
    list[dict[str, Any]],
]:
    """Retry only no-face results; never resolve a multi-face image by guessing."""
    original = embedder.embed_rgb(image_rgb)
    attempts = [_attempt_record("original", original)]
    if original.error != "no_face" or not padding_ratios:
        return original, "original", (0, 0, 0, 0), attempts

    selected = original
    selected_variant = "original"
    selected_padding = (0, 0, 0, 0)
    for ratio in padding_ratios:
        padded, padding = _padded_canvas(image_rgb, ratio)
        variant = f"neutral_pad_{ratio:g}_per_side"
        observation = embedder.embed_rgb(padded)
        attempts.append(_attempt_record(variant, observation))
        if observation.embedding is not None:
            left, top, _, _ = padding
            bbox = observation.bbox_xyxy
            translated_bbox = None
            if bbox is not None:
                translated_bbox = (
                    bbox[0] - left,
                    bbox[1] - top,
                    bbox[2] - left,
                    bbox[3] - top,
                )
            translated = EmbeddingObservation(
                observation.face_count,
                observation.embedding,
                observation.detection_score,
                translated_bbox,
                observation.error,
            )
            return translated, variant, padding, attempts
        if observation.error != "no_face":
            return observation, variant, padding, attempts
    return selected, selected_variant, selected_padding, attempts


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
    retry_padding_ratios: tuple[float, ...] = (),
    overwrite: bool = False,
) -> dict[str, Any]:
    retry_padding_ratios = _normalized_padding_ratios(retry_padding_ratios)
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
    selected_input_variants: list[str] = []
    metadata: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    embedded_variant_counts: Counter[str] = Counter()

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
            observation, selected_variant, padding, attempts = _embed_with_padding_retries(
                embedder, image_rgb, retry_padding_ratios
            )
            status = "embedded" if observation.error is None else "rejected"
            error = observation.error or ""
        except Exception as exc:  # per-file quarantine; type is recorded, not biometric content
            observation = None
            selected_variant = ""
            padding = (0, 0, 0, 0)
            attempts = []
            status = "rejected"
            error = type(exc).__name__

        embedding_row: int | str = ""
        if observation is not None and observation.embedding is not None:
            embedding_row = len(vectors)
            vectors.append(observation.embedding)
            person_ids.append(row["candidate_person_id"])
            relative_paths.append(relative_path)
            source_hashes.append(row["sha256"])
            selected_input_variants.append(selected_variant)
            embedded_variant_counts[selected_variant] += 1
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
                "selected_input_variant": selected_variant,
                "padding_ltrb": json.dumps(padding),
                "detection_attempts_json": json.dumps(attempts, separators=(",", ":")),
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
        selected_input_variants=np.asarray(selected_input_variants, dtype=np.str_),
        model_pack=np.asarray([pack], dtype=np.str_),
        model_hashes_json=np.asarray([json.dumps(model_hashes, sort_keys=True)], dtype=np.str_),
    )
    _atomic_csv(metadata_path, metadata)
    report = {
        "schema_version": 2,
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
        "retry_padding_ratios": list(retry_padding_ratios),
        "rescued_samples": sum(
            count
            for variant, count in embedded_variant_counts.items()
            if variant != "original"
        ),
        "embedded_by_input_variant": dict(sorted(embedded_variant_counts.items())),
        "outcomes": dict(sorted(outcome_counts.items())),
        "embeddings_sha256": sha256_file(embeddings_path),
        "metadata_sha256": sha256_file(metadata_path),
        "privacy": "restricted biometric; do not publish",
    }
    _atomic_json(report_path, report)
    return report
