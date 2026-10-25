"""Strict validation for immutable recognition enrollment and probe manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .config import resolve_under, sha256_file


class RecognitionManifestError(RuntimeError):
    """Recognition split manifests violate the frozen evaluation contract."""


FIELDS = {
    "sample_id",
    "source_dataset",
    "relative_path",
    "verified_person_id",
    "session_id",
    "device_id",
    "source_sha256",
    "permission_id",
}


@dataclass(frozen=True)
class RecognitionRecord:
    sample_id: str
    source_dataset: str
    relative_path: str
    person_id: str
    session_id: str
    device_id: str
    source_sha256: str
    permission_id: str
    split: str


def _read(path: Path, split: str) -> list[RecognitionRecord]:
    if not path.is_file():
        raise RecognitionManifestError(f"Manifest is absent: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != FIELDS:
            raise RecognitionManifestError(
                f"{path.name} fields mismatch; required exactly {sorted(FIELDS)}"
            )
        rows = list(reader)
    if not rows:
        raise RecognitionManifestError(f"Manifest is empty: {path}")
    records: list[RecognitionRecord] = []
    for index, row in enumerate(rows, start=2):
        required = (
            "sample_id",
            "source_dataset",
            "relative_path",
            "verified_person_id",
            "source_sha256",
            "permission_id",
        )
        if any(not row[field].strip() for field in required):
            raise RecognitionManifestError(f"{path.name}:{index} has an empty required field")
        source_hash = row["source_sha256"].lower()
        if len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
            raise RecognitionManifestError(f"{path.name}:{index} has an invalid source hash")
        records.append(
            RecognitionRecord(
                sample_id=row["sample_id"],
                source_dataset=row["source_dataset"],
                relative_path=row["relative_path"],
                person_id=row["verified_person_id"],
                session_id=row["session_id"],
                device_id=row["device_id"],
                source_sha256=source_hash,
                permission_id=row["permission_id"],
                split=split,
            )
        )
    return records


def validate_recognition_manifests(
    *,
    dataset_root: str | Path,
    manifests: dict[str, str | Path],
    allow_unknown_session_metadata: bool,
) -> dict[str, object]:
    expected_splits = {
        "enrollment",
        "val_known",
        "val_impostor",
        "test_known",
        "test_impostor",
    }
    if set(manifests) != expected_splits:
        raise RecognitionManifestError("Exactly five named recognition manifests are required")
    records = {
        split: _read(Path(path), split) for split, path in manifests.items()
    }
    all_rows = [row for rows in records.values() for row in rows]
    sample_ids = [row.sample_id for row in all_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RecognitionManifestError("Sample IDs must be globally unique")
    hashes = [row.source_sha256 for row in all_rows]
    if len(hashes) != len(set(hashes)):
        raise RecognitionManifestError("A source file hash crosses recognition splits")
    paths = [row.relative_path for row in all_rows]
    if len(paths) != len(set(paths)):
        raise RecognitionManifestError("A relative path crosses recognition splits")
    if not allow_unknown_session_metadata and any(
        not row.session_id or not row.device_id for row in all_rows
    ):
        raise RecognitionManifestError(
            "Session/device metadata is incomplete; explicitly allow the frozen limitation"
        )
    enrollment_ids = {row.person_id for row in records["enrollment"]}
    if len(enrollment_ids) < 2:
        raise RecognitionManifestError("Enrollment requires at least two identities")
    for split in ("val_known", "test_known"):
        unknown = {row.person_id for row in records[split]}.difference(enrollment_ids)
        if unknown:
            raise RecognitionManifestError(
                f"{split} contains identities absent from enrollment: {sorted(unknown)}"
            )
    for split in ("val_impostor", "test_impostor"):
        overlap = {row.person_id for row in records[split]}.intersection(enrollment_ids)
        if overlap:
            raise RecognitionManifestError(
                f"{split} identities overlap enrollment: {sorted(overlap)}"
            )
    root = Path(dataset_root)
    for row in all_rows:
        try:
            image_path = resolve_under(root, row.relative_path)
        except Exception as exc:
            raise RecognitionManifestError(
                f"Invalid source path in {row.split}: {row.relative_path}"
            ) from exc
        if sha256_file(image_path) != row.source_sha256:
            raise RecognitionManifestError(f"Source hash mismatch: {row.relative_path}")
        try:
            with Image.open(image_path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            raise RecognitionManifestError(
                f"Source image cannot be decoded: {row.relative_path}"
            ) from exc
    return {
        "schema_version": 1,
        "counts": {split: len(rows) for split, rows in records.items()},
        "enrollment_identities": len(enrollment_ids),
        "total_samples": len(all_rows),
        "unknown_session_metadata_allowed": allow_unknown_session_metadata,
    }
