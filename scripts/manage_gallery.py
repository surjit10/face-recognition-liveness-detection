#!/usr/bin/env python3
"""Manage persistent on-disk biometric gallery (list, inspect, delete, verify)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.gallery import Gallery, GalleryError  # noqa: E402


def get_manifest_sha256(gallery_path: Path) -> str:
    """Extract manifest_sha256 from gallery archive or fallback."""
    with np.load(gallery_path, allow_pickle=False) as archive:
        if "manifest_sha256" in archive.files:
            return str(archive["manifest_sha256"].reshape(-1)[0])
    return "0" * 64


def load_gallery_unverified(gallery_path: Path) -> tuple[Gallery, str]:
    """Load gallery by reading its self-declared embedding model hash."""
    if not gallery_path.is_file():
        raise GalleryError(f"Gallery file not found: {gallery_path}")
    with np.load(gallery_path, allow_pickle=False) as archive:
        model_hash = str(archive["embedding_model_sha256"].reshape(-1)[0])
    manifest_hash = get_manifest_sha256(gallery_path)
    current_hash = sha256_file(gallery_path)
    gallery = Gallery.load(
        gallery_path,
        expected_sha256=current_hash,
        expected_embedding_model_sha256=model_hash,
    )
    return gallery, manifest_hash


def list_identities(gallery_path: Path) -> dict:
    gallery, _ = load_gallery_unverified(gallery_path)
    summary = {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "identities_count": gallery.identity_count,
        "vectors_count": gallery.size,
        "identities": [
            {"person_id": person_id, "vectors": len(indices)}
            for person_id, indices in sorted(gallery._row_indices.items())
        ],
    }
    return summary


def inspect_identity(gallery_path: Path, person_id: str) -> dict:
    gallery, manifest_hash = load_gallery_unverified(gallery_path)
    if person_id not in gallery._row_indices:
        raise GalleryError(f"Person ID not found in gallery: {person_id}")
    indices = gallery._row_indices[person_id]
    hashes = [gallery._source_sha256[i] for i in indices if gallery._source_sha256[i]]
    return {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "person_id": person_id,
        "vectors_count": len(indices),
        "source_hashes_count": len(hashes),
        "manifest_sha256": manifest_hash,
    }


def delete_identity(
    gallery_path: Path,
    person_id: str,
    *,
    dry_run: bool = False,
    confirmed: bool = False,
) -> dict:
    gallery, manifest_hash = load_gallery_unverified(gallery_path)
    if person_id not in gallery._row_indices:
        raise GalleryError(f"Person ID not found in gallery: {person_id}")

    current_identities = gallery.identity_count
    current_vectors = gallery.size
    vectors_to_remove = len(gallery._row_indices[person_id])

    if current_identities <= 2:
        raise GalleryError("Deletion blocked: gallery must retain at least two identities.")

    if dry_run:
        return {
            "schema_version": 1,
            "status": "dry_run",
            "person_id": person_id,
            "vectors_to_remove": vectors_to_remove,
            "projected_identities": current_identities - 1,
            "projected_vectors": current_vectors - vectors_to_remove,
        }

    if not confirmed:
        raise GalleryError(
            f"Deletion requires explicit confirmation (--yes) for person ID: {person_id}"
        )

    json_path = gallery_path.with_suffix(".json")
    npz_backup = gallery_path.with_suffix(".npz.bak")
    json_backup = json_path.with_suffix(".json.bak")

    shutil.copy2(gallery_path, npz_backup)
    if json_path.is_file():
        shutil.copy2(json_path, json_backup)

    try:
        gallery.remove(person_id)
        new_gallery_hash = gallery.save(gallery_path, manifest_sha256=manifest_hash)

        if json_path.is_file():
            sidecar_data = json.loads(json_path.read_text(encoding="utf-8"))
            sidecar_data["gallery_sha256"] = new_gallery_hash
            sidecar_data["identities"] = gallery.identity_count
            sidecar_data["vectors"] = gallery.size
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=json_path.parent,
                prefix=f".{json_path.name}.",
                delete=False,
            ) as handle:
                json.dump(sidecar_data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temp_json = Path(handle.name)
            os.chmod(temp_json, 0o600)
            os.replace(temp_json, json_path)

        # Integrity validation
        Gallery.load(
            gallery_path,
            expected_sha256=new_gallery_hash,
            expected_embedding_model_sha256=gallery.embedding_model_sha256,
        )

        npz_backup.unlink(missing_ok=True)
        json_backup.unlink(missing_ok=True)

        return {
            "schema_version": 1,
            "status": "deleted",
            "person_id": person_id,
            "removed_vectors": vectors_to_remove,
            "remaining_identities": gallery.identity_count,
            "remaining_vectors": gallery.size,
            "new_gallery_sha256": new_gallery_hash,
            "note": "Active running instances must be restarted to reload the gallery from disk.",
        }
    except Exception as exc:
        if npz_backup.is_file():
            shutil.copy2(npz_backup, gallery_path)
            npz_backup.unlink(missing_ok=True)
        if json_backup.is_file():
            shutil.copy2(json_backup, json_path)
            json_backup.unlink(missing_ok=True)
        raise GalleryError(f"Deletion failed, restored original gallery: {exc}") from exc


def verify_gallery(gallery_path: Path) -> dict:
    gallery, manifest_hash = load_gallery_unverified(gallery_path)
    json_path = gallery_path.with_suffix(".json")
    sidecar_matches = None
    if json_path.is_file():
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        sidecar_matches = (
            sidecar.get("gallery_sha256") == sha256_file(gallery_path)
            and sidecar.get("identities") == gallery.identity_count
            and sidecar.get("vectors") == gallery.size
        )
    return {
        "schema_version": 1,
        "gallery_file": str(gallery_path),
        "identities": gallery.identity_count,
        "vectors": gallery.size,
        "gallery_sha256": sha256_file(gallery_path),
        "manifest_sha256": manifest_hash,
        "sidecar_consistent": sidecar_matches,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage persistent FaceGuard gallery.")
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=PROJECT_ROOT / "models/private/gallery/gallery.npz",
        help="Path to gallery.npz file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List enrolled identity IDs and vector counts")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific identity")
    inspect_parser.add_argument("--person-id", required=True, help="Person ID to inspect")

    delete_parser = subparsers.add_parser("delete", help="Delete a specific identity")
    delete_parser.add_argument("--person-id", required=True, help="Person ID to delete")
    delete_parser.add_argument(
        "--dry-run", action="store_true", help="Simulate deletion without modifying files"
    )
    delete_parser.add_argument(
        "--yes", action="store_true", help="Confirm deletion of the identity"
    )

    subparsers.add_parser("verify", help="Verify gallery integrity and sidecar consistency")

    retention_parser = subparsers.add_parser(
        "retention", help="Report biometric retention and expiry policy status"
    )
    retention_parser.add_argument("--days", type=int, default=365)

    args = parser.parse_args()

    try:
        if args.command == "list":
            result = list_identities(args.gallery_path)
        elif args.command == "inspect":
            result = inspect_identity(args.gallery_path, args.person_id)
        elif args.command == "delete":
            result = delete_identity(
                args.gallery_path,
                args.person_id,
                dry_run=args.dry_run,
                confirmed=args.yes,
            )
        elif args.command == "verify":
            result = verify_gallery(args.gallery_path)
        elif args.command == "retention":
            result = {
                "schema_version": 1,
                "status": "manual_retention_only",
                "message": (
                    "FRAD dataset metadata does not contain trusted enrollment timestamps. "
                    "Automatic expiry cannot be safely determined without fabricated dates. "
                    "Use 'delete --person-id <ID>' for manual removal."
                ),
            }
        else:
            parser.error(f"Unknown command: {args.command}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (GalleryError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
