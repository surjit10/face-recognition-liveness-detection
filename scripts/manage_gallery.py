#!/usr/bin/env python3
"""Manage the persistent biometric gallery through shared application APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.gallery import GalleryError  # noqa: E402
from faceguard.gallery_admin import (  # noqa: E402
    delete_identity,
    inspect_identity,
    list_identities,
    load_gallery_unverified,
    verify_gallery,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage persistent FaceGuard gallery.")
    parser.add_argument(
        "--gallery-path",
        type=Path,
        default=PROJECT_ROOT / "models/private/gallery/gallery.npz",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional runtime config whose pinned gallery hash is updated atomically",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List enrolled identity IDs and vector counts")
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific identity")
    inspect_parser.add_argument("--person-id", required=True)
    delete_parser = subparsers.add_parser("delete", help="Delete a specific identity")
    delete_parser.add_argument("--person-id", required=True)
    delete_parser.add_argument("--dry-run", action="store_true")
    delete_parser.add_argument("--yes", action="store_true")
    subparsers.add_parser("verify", help="Verify gallery integrity and sidecar consistency")
    retention_parser = subparsers.add_parser("retention", help="Report retention policy status")
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
                runtime_config_path=args.config,
            )
        elif args.command == "verify":
            result = verify_gallery(args.gallery_path)
        else:
            result = {
                "schema_version": 1,
                "status": "manual_retention_only",
                "message": (
                    "Legacy metadata has no trusted enrollment timestamps. Use "
                    "'delete --person-id <ID>' for manual removal."
                ),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (GalleryError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
