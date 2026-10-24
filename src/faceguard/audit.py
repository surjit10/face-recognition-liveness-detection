"""Non-destructive structural audit for folder-per-identity image datasets."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import sha256_file


INVENTORY_COLUMNS = (
    "schema_version",
    "candidate_person_id",
    "relative_path",
    "sha256",
    "size_bytes",
    "width",
    "height",
    "channels",
    "decode_ok",
    "face_count",
    "detect_score",
    "include",
    "exclusion_reason",
    "verified_person_id",
    "session_id",
    "device_id",
    "capture_time_source",
    "split",
    "permission_id",
)

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBJECT_COLUMNS = (
    "verified_person_id",
    "candidate_folder",
    "include",
    "image_count_discovered",
    "image_count_included",
    "session_count_verified",
    "permission_id",
    "decision_evidence",
)


class AuditError(RuntimeError):
    """Dataset audit cannot safely continue."""


def _atomic_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _candidate_files(root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for person_dir in sorted(root.iterdir(), key=lambda path: path.name):
        if not person_dir.is_dir() or person_dir.name.startswith("."):
            continue
        for image_path in sorted(person_dir.iterdir(), key=lambda path: path.name):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_SUFFIXES:
                candidates.append((person_dir.name, image_path))
    return candidates


def audit_dataset(
    dataset_root: str | Path,
    inventory_path: str | Path,
    subject_path: str | Path,
    report_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise AuditError(f"Dataset root is not a directory: {root}")
    inventory = Path(inventory_path)
    subjects = Path(subject_path)
    report = Path(report_path)
    for output in (inventory, subjects, report):
        if output.exists() and not overwrite:
            raise AuditError(f"Output exists; pass --overwrite to replace it: {output}")

    candidates = _candidate_files(root)
    if not candidates:
        raise AuditError(f"No supported images found under folder-per-person root: {root}")

    rows: list[dict[str, Any]] = []
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    counts_by_person: Counter[str] = Counter()
    decode_failures = 0

    for person_id, image_path in candidates:
        relative_path = image_path.relative_to(root).as_posix()
        file_hash = sha256_file(image_path)
        paths_by_hash[file_hash].append(relative_path)
        counts_by_person[person_id] += 1
        width: int | str = ""
        height: int | str = ""
        channels: int | str = ""
        decode_ok = False
        exclusion_reason = ""
        include: bool | str = "UNRESOLVED"
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                width, height = image.size
                channels = len(image.getbands())
                decode_ok = width > 0 and height > 0 and channels > 0
        except (OSError, ValueError, UnidentifiedImageError):
            decode_failures += 1
            exclusion_reason = "decode_failed"
            include = False

        rows.append(
            {
                "schema_version": 1,
                "candidate_person_id": person_id,
                "relative_path": relative_path,
                "sha256": file_hash,
                "size_bytes": image_path.stat().st_size,
                "width": width,
                "height": height,
                "channels": channels,
                "decode_ok": str(decode_ok).lower(),
                "face_count": "",
                "detect_score": "",
                "include": str(include).lower() if isinstance(include, bool) else include,
                "exclusion_reason": exclusion_reason,
                "verified_person_id": "",
                "session_id": "",
                "device_id": "",
                "capture_time_source": "",
                "split": "",
                "permission_id": "",
            }
        )

    row_by_path = {row["relative_path"]: row for row in rows}
    duplicate_groups = 0
    duplicate_files = 0
    for paths in paths_by_hash.values():
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        for duplicate_path in sorted(paths)[1:]:
            duplicate_files += 1
            row_by_path[duplicate_path]["include"] = "false"
            row_by_path[duplicate_path]["exclusion_reason"] = "exact_duplicate"

    rows.sort(key=lambda row: (row["candidate_person_id"], row["relative_path"]))
    subject_rows = [
        {
            "verified_person_id": "",
            "candidate_folder": person_id,
            "include": "UNRESOLVED",
            "image_count_discovered": count,
            "image_count_included": "",
            "session_count_verified": "",
            "permission_id": "",
            "decision_evidence": "",
        }
        for person_id, count in sorted(counts_by_person.items())
    ]
    _atomic_csv(inventory, rows, INVENTORY_COLUMNS)
    _atomic_csv(subjects, subject_rows, SUBJECT_COLUMNS)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root_name": root.name,
        "candidate_subjects": len(counts_by_person),
        "discovered_files": len(rows),
        "decoded_files": len(rows) - decode_failures,
        "decode_failures": decode_failures,
        "exact_duplicate_groups": duplicate_groups,
        "exact_duplicate_files_excluded": duplicate_files,
        "counts_by_person": dict(sorted(counts_by_person.items())),
        "inventory_path": inventory.as_posix(),
        "inventory_sha256": sha256_file(inventory),
        "subject_manifest_path": subjects.as_posix(),
        "subject_manifest_sha256": sha256_file(subjects),
        "limitations": [
            "No identity, consent, session, device, pose, or bona-fide semantics were inferred.",
            "face_count and detect_score remain unresolved until the frozen face model is run.",
            "UNRESOLVED include values require an authoritative cohort decision.",
        ],
    }
    _atomic_json(report, summary)
    return summary
