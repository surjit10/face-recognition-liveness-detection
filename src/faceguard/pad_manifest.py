"""Validated CelebA-Spoof manifest preparation for PAD training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageOps

from .config import sha256_file


class PadManifestError(RuntimeError):
    """PAD metadata is missing, ambiguous, or unsafe to use."""


DEFAULT_CROP_SCALE = 1.3
DEFAULT_MAX_CLIPPED_FRACTION = 0.2


@dataclass(frozen=True)
class PadRecord:
    relative_path: str
    subject_id: str
    source_split: str
    target_live: int
    source_live_spoof_label: int
    spoof_type: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class _PadCandidate:
    """A metadata-only record; creating one never touches an image or bbox file."""

    relative_path: str
    subject_id: str
    source_split: str
    target_live: int
    source_live_spoof_label: int
    spoof_type: int


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


def validate_pad_split_rows(
    split_rows: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, int]]:
    """Fail if train/validation/test reuse a subject, file hash, or path."""
    if set(split_rows) != {"train", "val", "test"}:
        raise PadManifestError("PAD validation requires train, val, and test splits")
    summary: dict[str, dict[str, int]] = {}
    subject_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    path_sets: dict[str, set[str]] = {}
    for name, rows in split_rows.items():
        if not rows:
            raise PadManifestError(f"PAD split {name} is empty")
        required = {"subject_id", "relative_path", "sha256", "target_live"}
        if required.difference(rows[0]):
            raise PadManifestError(f"PAD split {name} is missing leakage-check fields")
        subjects = {row["subject_id"].strip() for row in rows}
        paths = {row["relative_path"].strip() for row in rows}
        hashes = {row["sha256"].strip().lower() for row in rows}
        if "" in subjects or "" in paths or "" in hashes:
            raise PadManifestError(f"PAD split {name} contains an empty identity/path/hash")
        if len(paths) != len(rows) or len(hashes) != len(rows):
            raise PadManifestError(f"PAD split {name} contains duplicate paths or hashes")
        if any(len(value) != 64 for value in hashes):
            raise PadManifestError(f"PAD split {name} contains an invalid SHA-256")
        try:
            labels = {int(row["target_live"]) for row in rows}
        except ValueError as exc:
            raise PadManifestError(f"PAD split {name} has a non-integer label") from exc
        if labels != {0, 1}:
            raise PadManifestError(f"PAD split {name} must contain both PAD classes")
        subject_sets[name], path_sets[name], hash_sets[name] = subjects, paths, hashes
        summary[name] = {
            "rows": len(rows),
            "subjects": len(subjects),
            "bona_fide": sum(int(row["target_live"]) == 1 for row in rows),
            "attack": sum(int(row["target_live"]) == 0 for row in rows),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        subjects = subject_sets[left].intersection(subject_sets[right])
        if subjects:
            raise PadManifestError(
                f"PAD subject leakage between {left} and {right}: {sorted(subjects)[:10]}"
            )
        hashes = hash_sets[left].intersection(hash_sets[right])
        if hashes:
            raise PadManifestError(
                f"PAD file-hash leakage between {left} and {right}: {sorted(hashes)[:3]}"
            )
        paths = path_sets[left].intersection(path_sets[right])
        if paths:
            raise PadManifestError(
                f"PAD path leakage between {left} and {right}: {sorted(paths)[:3]}"
            )
    return summary


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
    if not all(math.isfinite(value) for value in bbox):
        raise PadManifestError(f"Bounding box contains non-finite values: {path}")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise PadManifestError(f"Bounding box has non-positive size: {path}")
    return bbox  # coordinates use the official 224-reference frame


def _crop_is_valid(
    image_size: tuple[int, int],
    bbox_224: tuple[float, float, float, float],
    crop_scale: float,
    max_clipped_fraction: float,
) -> bool:
    """Mirror train_pad.expanded_crop geometry without retaining image pixels."""

    width, height = image_size
    x, y, box_width, box_height = bbox_224
    x *= width / 224.0
    box_width *= width / 224.0
    y *= height / 224.0
    box_height *= height / 224.0
    side = crop_scale * max(box_width, box_height)
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    right = math.ceil(center_x + side / 2.0)
    bottom = math.ceil(center_y + side / 2.0)
    requested_area = max(1, right - left) * max(1, bottom - top)
    clipped_width = min(width, right) - max(0, left)
    clipped_height = min(height, bottom) - max(0, top)
    if clipped_width <= 0 or clipped_height <= 0:
        return False
    clipped_fraction = 1.0 - (clipped_width * clipped_height / requested_area)
    return clipped_fraction <= max_clipped_fraction


def _load_official_candidates(
    root: str | Path, source_split: str
) -> tuple[list[_PadCandidate], int]:
    """Load and validate label metadata without touching the dataset files."""

    if source_split not in {"train", "test"}:
        raise PadManifestError(f"Unsupported official split: {source_split}")
    dataset_root = Path(root).resolve(strict=True)
    metadata_path = _metadata_path(dataset_root, source_split)
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidates: list[_PadCandidate] = []
    skipped_count = 0
    for raw_path, labels in _label_items(document):
        try:
            relative_path = _safe_relative_path(raw_path)
            if len(labels) < 44:
                skipped_count += 1
                continue
            spoof_type = int(labels[40])
            source_live_spoof = int(labels[43])
            if source_live_spoof not in {0, 1}:
                skipped_count += 1
                continue

            lower_parts = {part.lower() for part in PurePosixPath(relative_path).parts}
            expected_source_label: int
            if "live" in lower_parts and "spoof" not in lower_parts:
                expected_source_label = 0
                if spoof_type != 0:
                    # Inconsistent CelebA-Spoof annotation (e.g. live folder but spoof_type > 0)
                    skipped_count += 1
                    continue
            elif "spoof" in lower_parts and "live" not in lower_parts:
                expected_source_label = 1
                if spoof_type == 0:
                    # Inconsistent CelebA-Spoof annotation (spoof folder but spoof_type == 0)
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
                continue

            if source_live_spoof != expected_source_label:
                skipped_count += 1
                continue

            subject_id = _subject_id(relative_path, source_split)
            candidates.append(
                _PadCandidate(
                    relative_path=relative_path,
                    subject_id=subject_id,
                    source_split=source_split,
                    target_live=1 - source_live_spoof,
                    source_live_spoof_label=source_live_spoof,
                    spoof_type=spoof_type,
                )
            )
        except Exception:
            skipped_count += 1
            continue

    if not candidates:
        raise PadManifestError(f"Official {source_split} metadata has no valid records")
    return candidates, skipped_count


def _validate_candidate(
    root: Path,
    candidate: _PadCandidate,
    *,
    crop_scale: float = DEFAULT_CROP_SCALE,
    max_clipped_fraction: float = DEFAULT_MAX_CLIPPED_FRACTION,
) -> PadRecord | None:
    """Decode one candidate and verify its bbox against the training crop contract."""

    try:
        image_path = root / candidate.relative_path
        if not image_path.is_file():
            return None
        bbox = _read_bbox(image_path)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            image_size = image.size
        if not _crop_is_valid(
            image_size,
            bbox,
            crop_scale,
            max_clipped_fraction,
        ):
            return None
    except Exception:
        return None
    return PadRecord(
        relative_path=candidate.relative_path,
        subject_id=candidate.subject_id,
        source_split=candidate.source_split,
        target_live=candidate.target_live,
        source_live_spoof_label=candidate.source_live_spoof_label,
        spoof_type=candidate.spoof_type,
        bbox=bbox,
    )


def load_official_records(root: str | Path, source_split: str) -> list[PadRecord]:
    """Load every valid record.

    This compatibility API intentionally validates the complete split. The optimized
    manifest builder below uses metadata candidates and stops at its requested quota.
    """

    dataset_root = Path(root).resolve(strict=True)
    candidates, skipped_count = _load_official_candidates(dataset_root, source_split)
    records = []
    for candidate in candidates:
        record = _validate_candidate(dataset_root, candidate)
        if record is None:
            skipped_count += 1
        else:
            records.append(record)

    if skipped_count > 0:
        print(
            f"[{source_split.upper()}] Filtered {skipped_count:,} noisy/missing samples; "
            f"kept {len(records):,} clean records.",
            flush=True,
        )

    if not records:
        raise PadManifestError(f"Official {source_split} metadata has no valid records")
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
        x, y, width, height = record.bbox
        digest = hashlib.sha256(
            f"{record.relative_path}:{record.subject_id}:{record.target_live}:{x:.1f}:{y:.1f}:{width:.1f}:{height:.1f}".encode("utf-8")
        ).hexdigest()
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


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _manifest_build_spec(
    root: Path,
    *,
    seed: int,
    validation_subject_fraction: float,
    train_limit: int | None,
    validation_limit: int | None,
    test_limit: int | None,
    validation_workers: int,
    crop_scale: float,
    max_clipped_fraction: float,
) -> tuple[dict[str, Any], str]:
    metadata_sha256 = {
        split: sha256_file(_metadata_path(root, split)) for split in ("train", "test")
    }
    spec = {
        "algorithm": "quota-first-v4-decode-and-crop-validation",
        "metadata_sha256": metadata_sha256,
        "seed": seed,
        "validation_subject_fraction": validation_subject_fraction,
        "validation_workers": validation_workers,
        "crop_scale": crop_scale,
        "max_clipped_fraction": max_clipped_fraction,
        "limits": {
            "train": train_limit,
            "val": validation_limit,
            "test": test_limit,
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return spec, fingerprint


def _cached_manifest_report(
    output: Path, build_fingerprint: str
) -> dict[str, Any] | None:
    report_path = output / "pad_split_report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("build_fingerprint") != build_fingerprint:
            return None
        for name in ("train", "val", "test"):
            manifest = output / f"pad_{name}.csv"
            if not manifest.is_file():
                return None
            if sha256_file(manifest) != report["manifest_sha256"][name]:
                return None
        # JSON object keys are strings; callers historically receive integer counters.
        for summary in report["splits"].values():
            for counter_name in ("target_live_counts", "spoof_type_counts"):
                summary[counter_name] = {
                    int(key): value for key, value in summary[counter_name].items()
                }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    print(
        f"[CACHE] Reusing verified PAD manifests from {output} "
        f"(fingerprint {build_fingerprint[:12]}).",
        flush=True,
    )
    return report


def _select_valid_records(
    root: Path,
    candidates: list[_PadCandidate],
    *,
    split_name: str,
    limit: int | None,
    seed: int,
    validation_workers: int,
    progress_every: int,
    crop_scale: float = DEFAULT_CROP_SCALE,
    max_clipped_fraction: float = DEFAULT_MAX_CLIPPED_FRACTION,
) -> list[PadRecord]:
    """Validate shuffled candidates in bounded batches and stop at the quota."""

    ordered = sorted(candidates, key=lambda row: row.relative_path)
    capped = limit is not None and limit > 0 and len(ordered) > limit
    desired = min(limit, len(ordered)) if capped and limit is not None else len(ordered)
    if desired == 0:
        raise PadManifestError(f"Split {split_name} has no metadata candidates")

    rng = random.Random(seed)
    by_label: dict[int, list[_PadCandidate]] = {0: [], 1: []}
    for candidate in ordered:
        by_label[candidate.target_live].append(candidate)
    for group in by_label.values():
        rng.shuffle(group)

    if capped:
        targets = {0: desired // 2, 1: desired - (desired // 2)}
    else:
        targets = {label: len(group) for label, group in by_label.items()}

    workers = max(1, validation_workers)
    batch_size = max(64, workers * 8)
    state: dict[str, Any] = {
        "checked": 0,
        "valid": 0,
        "rejected": 0,
        "next_report": max(1, progress_every),
        "last_report": time.monotonic(),
        "fail_fast_checked": False,
    }
    selected: list[PadRecord] = []
    checked_paths: set[str] = set()
    spare_records: list[PadRecord] = []
    stage_started = time.monotonic()

    print(
        f"[CPU/I-O][{split_name.upper()}] Metadata candidates: {len(ordered):,}; "
        f"validating up to {desired:,} samples with {workers} workers. "
        "GPU idle is expected during this stage.",
        flush=True,
    )

    def report_progress(force: bool = False) -> None:
        now = time.monotonic()
        if not force and state["checked"] < state["next_report"] and now - state["last_report"] < 10:
            return
        elapsed = max(now - stage_started, 1e-9)
        checked_rate = state["checked"] / elapsed
        selected_rate = len(selected) / elapsed
        remaining = max(0, desired - len(selected))
        eta = remaining / selected_rate if selected_rate > 0 else float("inf")
        percent = 100.0 * min(len(selected), desired) / desired
        eta_text = _format_duration(eta) if eta != float("inf") else "calculating"
        print(
            f"[CPU/I-O][{split_name.upper()}][ACTIVE] "
            f"{len(selected):,}/{desired:,} selected "
            f"({percent:5.1f}%) | {state['checked']:,} checked | "
            f"{state['rejected']:,} rejected | {checked_rate:,.1f} files/s | "
            f"elapsed {_format_duration(elapsed)} | ETA {eta_text}",
            flush=True,
        )
        state["next_report"] = state["checked"] + max(1, progress_every)
        state["last_report"] = now

    def validate_until(
        executor: ThreadPoolExecutor,
        pool: list[_PadCandidate],
        needed: int,
    ) -> None:
        cursor = 0
        accepted_here = 0
        while cursor < len(pool) and accepted_here < needed:
            batch = pool[cursor : cursor + batch_size]
            cursor += len(batch)
            results = executor.map(
                lambda row: _validate_candidate(
                    root,
                    row,
                    crop_scale=crop_scale,
                    max_clipped_fraction=max_clipped_fraction,
                ),
                batch,
            )
            for candidate, record in zip(batch, results):
                checked_paths.add(candidate.relative_path)
                state["checked"] += 1
                if record is None:
                    state["rejected"] += 1
                else:
                    state["valid"] += 1
                    if accepted_here < needed:
                        selected.append(record)
                        accepted_here += 1
                    else:
                        spare_records.append(record)

            if (
                not state["fail_fast_checked"]
                and state["checked"] >= 1_024
            ):
                state["fail_fast_checked"] = True
                valid_ratio = state["valid"] / state["checked"]
                if valid_ratio < 0.05:
                    raise PadManifestError(
                        f"Only {state['valid']:,}/{state['checked']:,} initial "
                        f"{split_name} candidates were valid ({valid_ratio:.1%}). "
                        "Check the dataset root and CelebA-Spoof file layout."
                    )
            report_progress()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pad-manifest") as executor:
        for label in (0, 1):
            validate_until(executor, by_label[label], targets[label])

        if len(selected) < desired:
            # A class may not contain enough usable records. Preserve the requested
            # total, when possible, using already-validated spares and unchecked rows.
            needed = desired - len(selected)
            selected.extend(spare_records[:needed])
            needed = desired - len(selected)
            if needed:
                fallback = [
                    candidate
                    for candidate in ordered
                    if candidate.relative_path not in checked_paths
                ]
                rng.shuffle(fallback)
                validate_until(executor, fallback, needed)

    if capped and len(selected) < desired:
        raise PadManifestError(
            f"Split {split_name} could supply only {len(selected):,}/{desired:,} "
            "valid samples after exhausting its metadata candidates"
        )
    if not selected:
        raise PadManifestError(f"Split {split_name} has no valid physical samples")
    report_progress(force=True)
    print(
        f"[CPU/I-O][{split_name.upper()}][COMPLETE] Selected {len(selected[:desired]):,} "
        f"samples in {_format_duration(time.monotonic() - stage_started)}.",
        flush=True,
    )
    return sorted(selected[:desired], key=lambda row: row.relative_path)


def prepare_celeba_spoof_manifests(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    validation_subject_fraction: float = 0.1,
    train_limit: int | None = 100_000,
    validation_limit: int | None = 20_000,
    test_limit: int | None = 40_000,
    validation_workers: int = 16,
    progress_every: int = 5_000,
    crop_scale: float = DEFAULT_CROP_SCALE,
    max_clipped_fraction: float = DEFAULT_MAX_CLIPPED_FRACTION,
    reuse_existing: bool = True,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if not 0.05 <= validation_subject_fraction <= 0.3:
        raise PadManifestError("validation_subject_fraction must be in [0.05, 0.3]")
    if validation_workers < 1:
        raise PadManifestError("validation_workers must be at least 1")
    if not math.isfinite(crop_scale) or crop_scale <= 0:
        raise PadManifestError("crop_scale must be a positive finite number")
    if not math.isfinite(max_clipped_fraction) or not 0 <= max_clipped_fraction < 1:
        raise PadManifestError("max_clipped_fraction must be in [0, 1)")

    build_spec, build_fingerprint = _manifest_build_spec(
        root,
        seed=seed,
        validation_subject_fraction=validation_subject_fraction,
        train_limit=train_limit,
        validation_limit=validation_limit,
        test_limit=test_limit,
        validation_workers=validation_workers,
        crop_scale=crop_scale,
        max_clipped_fraction=max_clipped_fraction,
    )
    if reuse_existing:
        cached = _cached_manifest_report(output, build_fingerprint)
        if cached is not None:
            return cached
    if output.exists() and any(output.iterdir()) and not overwrite_existing:
        raise PadManifestError(
            f"Output directory contains no reusable manifest cache: {output}. "
            "Use overwrite_existing=True to replace its managed manifest files."
        )

    official_train, train_metadata_skipped = _load_official_candidates(root, "train")
    official_test, test_metadata_skipped = _load_official_candidates(root, "test")
    print(
        f"[CPU][METADATA] Loaded {len(official_train):,} train and "
        f"{len(official_test):,} test candidates without scanning image files.",
        flush=True,
    )
    if train_metadata_skipped or test_metadata_skipped:
        print(
            f"[CPU][METADATA] Filtered inconsistent rows: train={train_metadata_skipped:,}, "
            f"test={test_metadata_skipped:,}.",
            flush=True,
        )

    # The official protocol is subject-disjoint, but some distributed mirrors contain
    # a small number of train rows whose identity directory also occurs in test. Keep
    # the official test split intact and quarantine those identities from all model
    # development data before deriving the train/validation split. Doing this at the
    # metadata stage also avoids an expensive physical-file scan that can only fail at
    # the final leakage check.
    test_subjects = {row.subject_id for row in official_test}
    overlapping_subjects = {
        row.subject_id for row in official_train if row.subject_id in test_subjects
    }
    overlapping_train_rows = sum(
        row.subject_id in overlapping_subjects for row in official_train
    )
    if overlapping_subjects:
        official_train = [
            row for row in official_train if row.subject_id not in overlapping_subjects
        ]
        print(
            f"[CPU][METADATA] Quarantined {overlapping_train_rows:,} train rows across "
            f"{len(overlapping_subjects):,} subject IDs also present in official test "
            "metadata; the official test rows were preserved.",
            flush=True,
        )

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
        "train": _select_valid_records(
            root,
            train_records,
            split_name="train",
            limit=train_limit,
            seed=seed + 1,
            validation_workers=validation_workers,
            progress_every=progress_every,
            crop_scale=crop_scale,
            max_clipped_fraction=max_clipped_fraction,
        ),
        "val": _select_valid_records(
            root,
            validation_records,
            split_name="val",
            limit=validation_limit,
            seed=seed + 2,
            validation_workers=validation_workers,
            progress_every=progress_every,
            crop_scale=crop_scale,
            max_clipped_fraction=max_clipped_fraction,
        ),
        "test": _select_valid_records(
            root,
            official_test,
            split_name="test",
            limit=test_limit,
            seed=seed + 3,
            validation_workers=validation_workers,
            progress_every=progress_every,
            crop_scale=crop_scale,
            max_clipped_fraction=max_clipped_fraction,
        ),
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
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "CelebA-Spoof official metadata",
        "dataset_root_name": root.name,
        "build_fingerprint": build_fingerprint,
        "build_spec": build_spec,
        "selection": {
            "strategy": "metadata-first balanced quota with bounded parallel image/crop validation",
            "validation_workers": validation_workers,
            "crop_scale": crop_scale,
            "max_clipped_fraction": max_clipped_fraction,
            "metadata_rows_filtered": {
                "train": train_metadata_skipped,
                "test": test_metadata_skipped,
            },
            "cross_split_quarantine": {
                "policy": "preserve official test; exclude overlapping subjects from train and validation",
                "subjects": len(overlapping_subjects),
                "train_rows": overlapping_train_rows,
            },
        },
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
