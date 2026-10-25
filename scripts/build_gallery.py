#!/usr/bin/env python3
"""Build a private gallery only from an explicit frozen enrollment manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.gallery import Gallery, GalleryError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-archive", type=Path, required=True)
    parser.add_argument("--enrollment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recognizer-file", default="w600k_r50.onnx")
    args = parser.parse_args()
    try:
        with args.enrollment_manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise GalleryError("Enrollment manifest is empty")
        required = {"relative_path", "verified_person_id", "source_sha256"}
        missing = required.difference(rows[0])
        if missing:
            raise GalleryError(f"Enrollment manifest fields are missing: {sorted(missing)}")
        selected: dict[str, tuple[str, str]] = {}
        for row in rows:
            relative_path = row["relative_path"].strip()
            person_id = row["verified_person_id"].strip()
            source_hash = row["source_sha256"].strip()
            if not relative_path or not person_id or len(source_hash) != 64:
                raise GalleryError("Enrollment rows require path, verified ID, and source hash")
            if relative_path in selected:
                raise GalleryError(f"Duplicate enrollment path: {relative_path}")
            selected[relative_path] = (person_id, source_hash)
        with np.load(args.embedding_archive, allow_pickle=False) as archive:
            needed = {
                "embeddings",
                "relative_paths",
                "source_sha256",
                "model_hashes_json",
            }
            missing = needed.difference(archive.files)
            if missing:
                raise GalleryError(f"Embedding archive fields are missing: {sorted(missing)}")
            vectors = archive["embeddings"]
            paths = archive["relative_paths"].astype(str)
            hashes = archive["source_sha256"].astype(str)
            model_hashes = json.loads(str(archive["model_hashes_json"].reshape(-1)[0]))
            recognizer_hash = model_hashes.get(args.recognizer_file)
            if not isinstance(recognizer_hash, str):
                raise GalleryError("Recognizer hash is absent from embedding metadata")
            by_path = {path: index for index, path in enumerate(paths.tolist())}
            missing_paths = sorted(set(selected).difference(by_path))
            if missing_paths:
                raise GalleryError(
                    f"Enrollment samples lack embeddings: {missing_paths[:5]}"
                )
            indices: list[int] = []
            person_ids: list[str] = []
            source_hashes: list[str] = []
            for path, (person_id, expected_hash) in selected.items():
                index = by_path[path]
                if hashes[index] != expected_hash:
                    raise GalleryError(f"Source hash mismatch for enrollment path: {path}")
                indices.append(index)
                person_ids.append(person_id)
                source_hashes.append(expected_hash)
            gallery = Gallery(
                vectors[indices],
                np.asarray(person_ids, dtype=np.str_),
                embedding_model_sha256=recognizer_hash,
                source_sha256=np.asarray(source_hashes, dtype=np.str_),
            )
        manifest_hash = sha256_file(args.enrollment_manifest)
        gallery_hash = gallery.save(args.output, manifest_sha256=manifest_hash)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, GalleryError) as exc:
        parser.exit(2, f"gallery build blocked: {exc}\n")
    report = {
        "schema_version": 1,
        "gallery_file": str(args.output),
        "gallery_sha256": gallery_hash,
        "gallery_manifest_sha256": manifest_hash,
        "embedding_model_sha256": recognizer_hash,
        "identities": gallery.identity_count,
        "vectors": gallery.size,
        "privacy": "restricted biometric; do not publish",
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
