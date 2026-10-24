"""Validated CelebA-Spoof manifest preparation for PAD training."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .config import sha256_file


class PadManifestError(RuntimeError):
    """PAD metadata is missing, ambiguous, or unsafe to use."""


@dataclass(frozen=True)
class PadRecord:
    relative_path: str
    subject_id: str
    source_split: str
    target_live: int
    source_live_spoof_label: int
    spoof_type: int
    bbox: tuple[float, float, float, float]


MANIFEST_COLUMNS = (
    "sample_id",
    "relative_path",
    "sha256",
    "subject_id",
    "source_split",
    "target_live",
    "source_live_spoof_label",
    "spoof_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
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


def _metadata_path(root: Path, split: str) -> Path:
    candidates = (
        root / "metas" / "intra_test" / f"{split}_label.json",
        root / "metas" / "labels" / f"{split}_label.json",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise PadManifestError(
            f"Expected exactly one official {split}_label.json under {root}; "
            f"found={existing}"
        )
    return existing[0]


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PadManifestError(f"Unsafe dataset-relative path: {value!r}")
    return path.as_posix()


def _label_items(document: Any) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(document, dict):
        for key, value in document.items():
            if isinstance(value, dict):
                relative_path = value.get("path", key)
                labels = value.get("labels")
            else:
                relative_path = key
                labels = value
            if not isinstance(relative_path, str) or not isinstance(labels, list):
                raise PadManifestError("Unsupported official label JSON entry")
            yield relative_path, labels
        return
    if isinstance(document, list):
        for value in document:
            if not isinstance(value, dict):
                raise PadManifestError("Unsupported official label JSON list entry")
            relative_path = value.get("path")
            labels = value.get("labels")
            if not isinstance(relative_path, str) or not isinstance(labels, list):
                raise PadManifestError("Label list entries require path and labels")
            yield relative_path, labels
        return
    raise PadManifestError("Official label JSON must be an object or a list")


def _subject_id(relative_path: str, source_split: str) -> str:
    parts = PurePosixPath(relative_path).parts
    lowered = [part.lower() for part in parts]
    try:
        split_index = lowered.index(source_split)
    except ValueError as exc:
        raise PadManifestError(
            f"Path does not contain its official split {source_split}: {relative_path}"
        ) from exc
    if split_index + 1 >= len(parts):
        raise PadManifestError(f"Path has no subject after split: {relative_path}")
    return parts[split_index + 1]


def _bbox_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_BB.txt")


def _read_bbox(image_path: Path) -> tuple[float, float, float, float]:
    path = _bbox_path(image_path)
    if not path.is_file():
        raise PadManifestError(f"Missing official bounding box: {path}")
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    if len(tokens) < 4:
        raise PadManifestError(f"Bounding box must contain x y w h: {path}")
    try:
        bbox = tuple(float(value) for value in tokens[:4])
    except ValueError as exc:
        raise PadManifestError(f"Invalid bounding box numbers: {path}") from exc
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise PadManifestError(f"Bounding box has non-positive size: {path}")
    return bbox  # coordinates use the official 224-reference frame


def load_official_records(root: str | Path, source_split: str) -> list[PadRecord]:
    if source_split not in {"train", "test"}:
        raise PadManifestError(f"Unsupported official split: {source_split}")
    dataset_root = Path(root).resolve(strict=True)
    metadata_path = _metadata_path(dataset_root, source_split)
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    records: list[PadRecord] = []
    for raw_path, labels in _label_items(document):
        relative_path = _safe_relative_path(raw_path)
        if len(labels) < 44:
            raise PadManifestError(
                f"Expected at least 44 official labels for {relative_path}; got {len(labels)}"
            )
        try:
            spoof_type = int(labels[40])
            source_live_spoof = int(labels[43])
        except (TypeError, ValueError) as exc:
            raise PadManifestError(f"Invalid PAD labels for {relative_path}") from exc
        if source_live_spoof not in {0, 1}:
            raise PadManifestError(f"Live/spoof label must be 0 or 1: {relative_path}")

        lower_parts = {part.lower() for part in PurePosixPath(relative_path).parts}
        expected_source_label: int
        if "live" in lower_parts and "spoof" not in lower_parts:
            expected_source_label = 0
            if spoof_type != 0:
                raise PadManifestError(
                    f"Live path has non-live spoof_type={spoof_type}: {relative_path}"
                )
        elif "spoof" in lower_parts and "live" not in lower_parts:
            expected_source_label = 1
            if spoof_type == 0:
                raise PadManifestError(
                    f"Spoof path has live spoof_type=0: {relative_path}"
                )
        else:
            raise PadManifestError(
                f"Path must contain exactly one official live/spoof component: {relative_path}"
            )
        if source_live_spoof != expected_source_label:
            raise PadManifestError(
                f"Official label/path disagreement for {relative_path}: "
                f"label={source_live_spoof}, expected={expected_source_label}"
            )

        image_path = dataset_root / relative_path
        if not image_path.is_file():
            raise PadManifestError(f"Labeled image is missing: {image_path}")
        records.append(
            PadRecord(
                relative_path=relative_path,
                subject_id=_subject_id(relative_path, source_split),
                source_split=source_split,
                target_live=1 - source_live_spoof,
                source_live_spoof_label=source_live_spoof,
                spoof_type=spoof_type,
                bbox=_read_bbox(image_path),
            )
        )
    if not records:
        raise PadManifestError(f"Official {source_split} metadata is empty")
    return records


def _balanced_cap(records: list[PadRecord], limit: int | None, seed: int) -> list[PadRecord]:
    ordered = sorted(records, key=lambda row: row.relative_path)
    if limit is None or limit <= 0 or len(ordered) <= limit:
        return ordered
    rng = random.Random(seed)
    by_label: dict[int, list[PadRecord]] = {0: [], 1: []}
    for record in ordered:
        by_label[record.target_live].append(record)
    target_each = limit // 2
    selected: list[PadRecord] = []
    for label in (0, 1):
        group = by_label[label]
        rng.shuffle(group)
        selected.extend(group[: min(target_each, len(group))])
    remaining = limit - len(selected)
    if remaining:
        selected_keys = {row.relative_path for row in selected}
        pool = [row for row in ordered if row.relative_path not in selected_keys]
        rng.shuffle(pool)
        selected.extend(pool[:remaining])
    return sorted(selected, key=lambda row: row.relative_path)


def _manifest_rows(root: Path, records: list[PadRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        digest = sha256_file(root / record.relative_path)
        x, y, width, height = record.bbox
        rows.append(
            {
                "sample_id": hashlib.sha256(
                    f"{record.relative_path}\0{digest}".encode("utf-8")
                ).hexdigest(),
                "relative_path": record.relative_path,
                "sha256": digest,
                "subject_id": record.subject_id,
                "source_split": record.source_split,
                "target_live": record.target_live,
                "source_live_spoof_label": record.source_live_spoof_label,
                "spoof_type": record.spoof_type,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": width,
                "bbox_h": height,
            }
        )
    return rows


def _summary(records: list[PadRecord]) -> dict[str, Any]:
    return {
        "samples": len(records),
        "subjects": len({row.subject_id for row in records}),
        "target_live_counts": dict(sorted(Counter(row.target_live for row in records).items())),
        "spoof_type_counts": dict(sorted(Counter(row.spoof_type for row in records).items())),
    }


def prepare_celeba_spoof_manifests(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    validation_subject_fraction: float = 0.1,
    train_limit: int | None = 100_000,
    validation_limit: int | None = 20_000,
    test_limit: int | None = 40_000,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise PadManifestError(f"Output directory is not empty: {output}")
    if not 0.05 <= validation_subject_fraction <= 0.3:
        raise PadManifestError("validation_subject_fraction must be in [0.05, 0.3]")

    official_train = load_official_records(root, "train")
    official_test = load_official_records(root, "test")
    train_subjects = sorted({row.subject_id for row in official_train})
    rng = random.Random(seed)
    rng.shuffle(train_subjects)
    validation_subject_count = max(
        1, int(round(len(train_subjects) * validation_subject_fraction))
    )
    validation_subjects = set(train_subjects[:validation_subject_count])
    train_records = [row for row in official_train if row.subject_id not in validation_subjects]
    validation_records = [row for row in official_train if row.subject_id in validation_subjects]

    split_records = {
        "train": _balanced_cap(train_records, train_limit, seed + 1),
        "val": _balanced_cap(validation_records, validation_limit, seed + 2),
        "test": _balanced_cap(official_test, test_limit, seed + 3),
    }
    subject_sets = {
        name: {row.subject_id for row in records} for name, records in split_records.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = subject_sets[left].intersection(subject_sets[right])
        if overlap:
            raise PadManifestError(
                f"Subject leakage between {left} and {right}: {sorted(overlap)[:10]}"
            )
    path_sets = {
        name: {row.relative_path for row in records} for name, records in split_records.items()
    }
    if path_sets["train"] & path_sets["val"] or path_sets["train"] & path_sets["test"] or path_sets["val"] & path_sets["test"]:
        raise PadManifestError("Image path leakage exists across PAD splits")
    for name, records in split_records.items():
        if {row.target_live for row in records} != {0, 1}:
            raise PadManifestError(f"Split {name} does not contain both PAD classes")

    output.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for name, records in split_records.items():
        path = output / f"pad_{name}.csv"
        _atomic_csv(path, _manifest_rows(root, records))
        manifest_paths[name] = path

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CelebA-Spoof official metadata",
        "dataset_root_name": root.name,
        "label_mapping": {"target_live_1": "bona_fide", "target_live_0": "attack"},
        "source_label_mapping": {"0": "live", "1": "spoof"},
        "seed": seed,
        "validation_subject_fraction": validation_subject_fraction,
        "limits": {
            "train": train_limit,
            "val": validation_limit,
            "test": test_limit,
        },
        "splits": {name: _summary(records) for name, records in split_records.items()},
        "manifest_sha256": {
            name: sha256_file(path) for name, path in manifest_paths.items()
        },
        "privacy": "restricted dataset metadata; do not publish image paths without review",
    }
    _atomic_json(output / "pad_split_report.json", report)
    return report
