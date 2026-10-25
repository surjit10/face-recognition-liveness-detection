"""Deterministic exploratory recognition split generation from audited embeddings."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .config import sha256_file


class RecognitionSplitError(RuntimeError):
    """The exploratory recognition split cannot be generated safely."""


MANIFEST_FIELDS = (
    "sample_id",
    "source_dataset",
    "relative_path",
    "verified_person_id",
    "session_id",
    "device_id",
    "source_sha256",
    "permission_id",
)


def _stable_key(seed: int, *values: str) -> str:
    payload = "|".join((str(seed), *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _probe_counts(sample_count: int, fraction: float) -> tuple[int, int]:
    validation = max(1, int(round(sample_count * fraction)))
    test = max(1, int(round(sample_count * fraction)))
    while sample_count - validation - test < 2:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            break
    if sample_count - validation - test < 2:
        raise RecognitionSplitError(
            f"Identity with {sample_count} embeddings cannot support enrollment/val/test"
        )
    return validation, test


def create_exploratory_recognition_split(
    *,
    embedding_archive: str | Path,
    output_dir: str | Path,
    seed: int,
    unknown_identity_fraction: float = 0.2,
    known_probe_fraction: float = 0.2,
    source_dataset: str = "frad_legacy_crops_arcface_audit",
    permission_id: str = "FRAD_PERMISSION",
    overwrite: bool = False,
) -> dict[str, Any]:
    if not isinstance(seed, int) or seed < 0:
        raise RecognitionSplitError("Seed must be a non-negative integer")
    if not 0.05 <= unknown_identity_fraction <= 0.5:
        raise RecognitionSplitError("Unknown identity fraction must be within [0.05, 0.5]")
    if not 0.05 <= known_probe_fraction <= 0.4:
        raise RecognitionSplitError("Known probe fraction must be within [0.05, 0.4]")
    archive_path = Path(embedding_archive)
    output = Path(output_dir)
    names = (
        "recognition_enrollment",
        "recognition_val_known",
        "recognition_val_impostor",
        "recognition_test_known",
        "recognition_test_impostor",
    )
    paths = {name: output / f"{name}.csv" for name in names}
    report_path = output / "recognition_split_summary.json"
    existing = [path for path in (*paths.values(), report_path) if path.exists()]
    if existing and not overwrite:
        raise RecognitionSplitError(f"Outputs already exist; pass overwrite: {existing}")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {
                "embeddings",
                "person_ids",
                "relative_paths",
                "source_sha256",
                "model_hashes_json",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RecognitionSplitError(
                    f"Embedding archive fields are missing: {sorted(missing)}"
                )
            embeddings = archive["embeddings"]
            person_ids = archive["person_ids"].astype(str)
            relative_paths = archive["relative_paths"].astype(str)
            source_hashes = archive["source_sha256"].astype(str)
            model_hashes = json.loads(str(archive["model_hashes_json"].reshape(-1)[0]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RecognitionSplitError(f"Unable to read embedding archive: {exc}") from exc
    count = len(embeddings)
    if not (len(person_ids) == len(relative_paths) == len(source_hashes) == count):
        raise RecognitionSplitError("Embedding archive arrays have inconsistent lengths")
    if len(set(source_hashes.tolist())) != count or len(set(relative_paths.tolist())) != count:
        raise RecognitionSplitError("Embedding archive contains duplicate paths or source hashes")
    by_identity: dict[str, list[int]] = {}
    for index, (person_id, relative_path, source_hash) in enumerate(
        zip(person_ids.tolist(), relative_paths.tolist(), source_hashes.tolist())
    ):
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise RecognitionSplitError(f"Unsafe embedding path: {relative_path}")
        if pure.parts[0] != person_id:
            raise RecognitionSplitError(
                f"Path identity {pure.parts[0]!r} does not match label {person_id!r}"
            )
        if len(source_hash) != 64:
            raise RecognitionSplitError(f"Invalid source hash for {relative_path}")
        by_identity.setdefault(person_id, []).append(index)
    identities = sorted(by_identity, key=lambda value: _stable_key(seed, "identity", value))
    if len(identities) < 10:
        raise RecognitionSplitError("At least ten identities are required for open-set splitting")
    unknown_count = max(2, int(round(len(identities) * unknown_identity_fraction)))
    unknown_identities = identities[:unknown_count]
    known_identities = identities[unknown_count:]
    validation_unknown_count = unknown_count // 2
    validation_unknown = set(unknown_identities[:validation_unknown_count])
    test_unknown = set(unknown_identities[validation_unknown_count:])
    if not validation_unknown or not test_unknown:
        raise RecognitionSplitError("Both validation and test need unknown identities")

    rows: dict[str, list[dict[str, str]]] = {name: [] for name in names}

    def append(split: str, index: int) -> None:
        person_id = str(person_ids[index])
        source_hash = str(source_hashes[index])
        rows[split].append(
            {
                "sample_id": _stable_key(seed, split, source_hash)[:24],
                "source_dataset": source_dataset,
                "relative_path": str(relative_paths[index]),
                "verified_person_id": person_id,
                "session_id": "",
                "device_id": "",
                "source_sha256": source_hash,
                "permission_id": permission_id,
            }
        )

    for person_id in known_identities:
        indices = sorted(
            by_identity[person_id],
            key=lambda index: _stable_key(
                seed, "sample", person_id, str(source_hashes[index])
            ),
        )
        validation_count, test_count = _probe_counts(len(indices), known_probe_fraction)
        for index in indices[:validation_count]:
            append("recognition_val_known", index)
        for index in indices[validation_count : validation_count + test_count]:
            append("recognition_test_known", index)
        for index in indices[validation_count + test_count :]:
            append("recognition_enrollment", index)
    for person_id in unknown_identities:
        destination = (
            "recognition_val_impostor"
            if person_id in validation_unknown
            else "recognition_test_impostor"
        )
        for index in sorted(
            by_identity[person_id],
            key=lambda value: _stable_key(seed, "unknown", str(source_hashes[value])),
        ):
            append(destination, index)
    for split_rows in rows.values():
        split_rows.sort(key=lambda row: (row["verified_person_id"], row["sample_id"]))
    for name, path in paths.items():
        _atomic_csv(path, rows[name])
    manifest_hashes = {name: sha256_file(path) for name, path in paths.items()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "exploratory_unknown_session_open_set_v1",
        "official_benchmark": False,
        "seed": seed,
        "source_embedding_archive": str(archive_path),
        "source_embedding_archive_sha256": sha256_file(archive_path),
        "embedding_model_hashes": model_hashes,
        "parameters": {
            "unknown_identity_fraction": unknown_identity_fraction,
            "known_probe_fraction_per_split": known_probe_fraction,
            "identity_order": "sha256(seed|identity|person_id)",
            "sample_order": "sha256(seed|sample|person_id|source_sha256)",
        },
        "identity_labels": {
            "count": len(identities),
            "preserved_without_renumbering": True,
            "contains_label_000": "000" in by_identity,
            "known_enrollment": sorted(known_identities),
            "validation_impostor": sorted(validation_unknown),
            "test_impostor": sorted(test_unknown),
        },
        "counts": {
            name: {
                "samples": len(split_rows),
                "identities": len({row["verified_person_id"] for row in split_rows}),
            }
            for name, split_rows in rows.items()
        },
        "manifest_sha256": manifest_hashes,
        "session_id_policy": "unknown; empty field; never inferred from filenames",
        "device_id_policy": "unknown; empty field; never inferred from filenames",
        "limitations": [
            "Exploratory protocol only; not an official FRAD benchmark.",
            "Known-person files are hash-disjoint, but cross-session independence cannot be claimed.",
            "Validation and test impostor identities are disjoint from enrollment and each other.",
        ],
    }
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report
