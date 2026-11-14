#!/usr/bin/env python3
"""Apply the runtime face-quality gate to the frozen recognition split manifests.

The original embedding extraction accepted any image in which exactly one face was
detected. In crowded scenes that single detection was sometimes a small background
bystander (about 20 px wide, detection score about 0.5) rather than the subject, which
put wrong-person vectors into the gallery and into the probe sets. The live
application never accepts such faces because it enforces `face_quality_reason`.

This script applies that same function, with the thresholds from
`configs/runtime.yaml`, to every recorded detection. Failing samples are removed from
the split manifests and every removal is recorded with its reason. Identity
assignments are not changed and nothing is re-split. Known-person probes whose
identity has no remaining enrollment vector are also removed, because the protocol
defines known probes as probes of enrolled identities.

Private inputs (embedding metadata and face images) are read locally only; the
outputs contain pseudonymous IDs, hashes and reasons, never pixels or embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.pipeline import face_quality_reason  # noqa: E402
from faceguard.runtime_config import load_runtime_config  # noqa: E402

SPLITS = (
    "recognition_enrollment",
    "recognition_val_known",
    "recognition_val_impostor",
    "recognition_test_known",
    "recognition_test_impostor",
)
KNOWN_SPLITS = {"recognition_val_known", "recognition_test_known"}
EXCLUSION_FIELDS = (
    "sample_id",
    "split",
    "verified_person_id",
    "source_sha256",
    "reason",
    "detection_score",
    "face_min_side_px",
)
NO_ENROLLMENT_REASON = "identity_has_no_quality_enrollment"


class QualityGateError(RuntimeError):
    """The manifests, metadata or images are inconsistent."""


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def _write_csv(path: Path, fields: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def gate_rows(
    split_rows: dict[str, list[dict[str, str]]],
    metadata: dict[str, dict[str, str]],
    load_image,
    policy,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, object]]]:
    """Return (kept rows per split, exclusion records). Pure apart from `load_image`."""
    kept: dict[str, list[dict[str, str]]] = {}
    exclusions: list[dict[str, object]] = []
    for split in SPLITS:
        kept[split] = []
        for row in split_rows[split]:
            record = metadata.get(row["relative_path"])
            if record is None or record.get("status") != "embedded":
                raise QualityGateError(f"No embedded detection for {row['sample_id']}")
            if record["source_sha256"] != row["source_sha256"]:
                raise QualityGateError(f"Source hash mismatch for {row['sample_id']}")
            score = float(record["detection_score"])
            bbox = tuple(float(value) for value in json.loads(record["bbox_xyxy"]))
            if len(bbox) != 4:
                raise QualityGateError(f"Invalid face box for {row['sample_id']}")
            image = load_image(row["relative_path"])
            reason = (
                "image_unreadable"
                if image is None
                else face_quality_reason(image, score, bbox, policy)  # type: ignore[arg-type]
            )
            if reason is None:
                kept[split].append(row)
                continue
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "verified_person_id": row["verified_person_id"],
                    "source_sha256": row["source_sha256"],
                    "reason": reason,
                    "detection_score": f"{score:.4f}",
                    "face_min_side_px": f"{min(bbox[2] - bbox[0], bbox[3] - bbox[1]):.1f}",
                }
            )
    enrolled = {row["verified_person_id"] for row in kept["recognition_enrollment"]}
    for split in KNOWN_SPLITS:
        still_kept = []
        for row in kept[split]:
            if row["verified_person_id"] in enrolled:
                still_kept.append(row)
                continue
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "verified_person_id": row["verified_person_id"],
                    "source_sha256": row["source_sha256"],
                    "reason": NO_ENROLLMENT_REASON,
                    "detection_score": "",
                    "face_min_side_px": "",
                }
            )
        kept[split] = still_kept
    return kept, exclusions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "data/manifests")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    parser.add_argument(
        "--exclusions-output",
        type=Path,
        default=PROJECT_ROOT / "data/manifests/recognition_quality_exclusions.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        policy = load_runtime_config(args.config, project_root=PROJECT_ROOT).pipeline_policy
        _, metadata_rows = _read_csv(args.embedding_metadata)
        metadata = {row["relative_path"]: row for row in metadata_rows}
        fields: dict[str, list[str]] = {}
        split_rows: dict[str, list[dict[str, str]]] = {}
        for split in SPLITS:
            fields[split], split_rows[split] = _read_csv(args.manifest_dir / f"{split}.csv")

        def load_image(relative_path: str):
            return cv2.imread(str(args.image_root / relative_path), cv2.IMREAD_COLOR)

        kept, exclusions = gate_rows(split_rows, metadata, load_image, policy)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, QualityGateError) as exc:
        parser.exit(2, f"quality gate blocked: {exc}\n")

    summary = {
        "excluded_by_split": dict(Counter(str(row["split"]) for row in exclusions)),
        "excluded_by_reason": dict(Counter(str(row["reason"]) for row in exclusions)),
        "remaining": {
            split: {
                "samples": len(kept[split]),
                "identities": len({row["verified_person_id"] for row in kept[split]}),
            }
            for split in SPLITS
        },
    }
    if args.exclusions_output.exists() and exclusions:
        parser.exit(
            2,
            "quality gate blocked: an exclusion record already exists but the manifests "
            "still contain failing samples; review manually\n",
        )
    if not exclusions:
        summary["status"] = "already_applied_or_nothing_to_exclude"
    elif not args.dry_run:
        for split in SPLITS:
            _write_csv(args.manifest_dir / f"{split}.csv", fields[split], kept[split])
        _write_csv(args.exclusions_output, EXCLUSION_FIELDS, exclusions)
        summary["status"] = "applied"
    else:
        summary["status"] = "dry_run"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
