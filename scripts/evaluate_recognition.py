#!/usr/bin/env python3
"""Create private validation/test gallery-search scores from frozen probe manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.gallery import Gallery, GalleryError  # noqa: E402


OUTPUT_FIELDS = (
    "probe_id",
    "probe_kind",
    "top1_person_id",
    "top1_score",
    "top1_top2_margin",
    "correct_identity",
)


def _read_manifest(path: Path, kind: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "relative_path", "verified_person_id", "source_sha256"}
    if not rows or required.difference(rows[0]):
        raise GalleryError(f"{path} is empty or missing fields: {sorted(required)}")
    for row in rows:
        row["probe_kind"] = kind
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-archive", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--gallery-report", type=Path, required=True)
    parser.add_argument("--known-manifest", type=Path, required=True)
    parser.add_argument("--impostor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        gallery_report = json.loads(args.gallery_report.read_text(encoding="utf-8"))
        gallery = Gallery.load(
            args.gallery,
            expected_sha256=gallery_report["gallery_sha256"],
            expected_embedding_model_sha256=gallery_report["embedding_model_sha256"],
        )
        probes = _read_manifest(args.known_manifest, "known") + _read_manifest(
            args.impostor_manifest, "impostor"
        )
        with np.load(args.embedding_archive, allow_pickle=False) as archive:
            required = {"embeddings", "relative_paths", "source_sha256"}
            missing = required.difference(archive.files)
            if missing:
                raise GalleryError(f"Embedding archive fields are missing: {sorted(missing)}")
            embeddings = archive["embeddings"]
            paths = archive["relative_paths"].astype(str)
            hashes = archive["source_sha256"].astype(str)
            by_path = {path: index for index, path in enumerate(paths.tolist())}
            output_rows: list[dict[str, object]] = []
            for probe in probes:
                path = probe["relative_path"]
                if path not in by_path:
                    raise GalleryError(f"Probe has no embedding: {path}")
                index = by_path[path]
                if hashes[index] != probe["source_sha256"]:
                    raise GalleryError(f"Probe source hash mismatch: {path}")
                ranking = gallery.rank(embeddings[index])
                correct = (
                    probe["probe_kind"] == "known"
                    and ranking.top1_person_id == probe["verified_person_id"]
                    and not ranking.tied
                )
                output_rows.append(
                    {
                        "probe_id": probe["sample_id"],
                        "probe_kind": probe["probe_kind"],
                        "top1_person_id": ranking.top1_person_id,
                        "top1_score": ranking.top1_score,
                        "top1_top2_margin": ranking.margin,
                        "correct_identity": str(correct).lower(),
                    }
                )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, GalleryError) as exc:
        parser.exit(2, f"recognition evaluation blocked: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=args.output.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "known_probes": sum(row["probe_kind"] == "known" for row in output_rows),
                "impostor_probes": sum(
                    row["probe_kind"] == "impostor" for row in output_rows
                ),
                "privacy": "private search scores; no embeddings included",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
