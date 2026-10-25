#!/usr/bin/env python3
"""Atomic model release management, switching, and rollback."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.runtime_config import load_runtime_config  # noqa: E402
import generate_release_manifest as GRM  # noqa: E402


class ReleaseError(RuntimeError):
    """Raised when release registration, switching, or rollback fails."""


@contextmanager
def release_lock(releases_dir: Path) -> Iterator[None]:
    releases_dir.mkdir(parents=True, exist_ok=True)
    lock_file = releases_dir / ".release.lock"
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        raise ReleaseError(
            f"Release operation is already in progress (locked by {lock_file})"
        )
    try:
        yield
    finally:
        try:
            lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_symlink(target_dir: Path, link_path: Path) -> None:
    temp_link = link_path.parent / f".tmp_{link_path.name}_{os.getpid()}"
    try:
        temp_link.unlink(missing_ok=True)
        # Create relative symlink if within same parent
        if target_dir.parent == link_path.parent:
            os.symlink(target_dir.name, temp_link)
        else:
            os.symlink(target_dir, temp_link)
        os.replace(temp_link, link_path)
    finally:
        temp_link.unlink(missing_ok=True)


def register_release(
    config_path: Path,
    releases_dir: Path = PROJECT_ROOT / "releases",
    project_root: Path = PROJECT_ROOT,
) -> dict:
    with release_lock(releases_dir):
        manifest = GRM.build_manifest(config_path, project_root=project_root)
        release_id = manifest["release_id"]
        release_dir = releases_dir / release_id
        release_dir.mkdir(parents=True, exist_ok=True)

        dest_manifest = release_dir / "release_manifest.json"
        dest_config = release_dir / "runtime.yaml"

        GRM.write_manifest(manifest, dest_manifest)
        shutil.copy2(config_path, dest_config)

        return {
            "schema_version": 1,
            "status": "registered",
            "release_id": release_id,
            "release_dir": str(release_dir),
            "manifest_file": str(dest_manifest),
        }


def switch_release(
    release_id: str,
    releases_dir: Path = PROJECT_ROOT / "releases",
    project_root: Path = PROJECT_ROOT,
) -> dict:
    with release_lock(releases_dir):
        target_dir = releases_dir / release_id
        if not target_dir.is_dir():
            raise ReleaseError(f"Release {release_id} is not registered in {releases_dir}")

        manifest_file = target_dir / "release_manifest.json"
        config_file = target_dir / "runtime.yaml"

        if not manifest_file.is_file() or not config_file.is_file():
            raise ReleaseError(f"Release {release_id} is incomplete (missing manifest or config)")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        # Verify artifact integrity before activating
        try:
            GRM.validate_manifest(manifest, project_root=project_root)
        except GRM.ReleaseManifestError as exc:
            raise ReleaseError(f"Validation failed for release {release_id}: {exc}") from exc

        # Health validation: check runtime config parses without error
        try:
            load_runtime_config(config_file, project_root=project_root)
        except Exception as exc:
            raise ReleaseError(f"Health verification failed for release {release_id}: {exc}") from exc

        active_link = releases_dir / "active"
        previous_link = releases_dir / "previous"

        current_active = None
        if active_link.is_symlink() or active_link.exists():
            try:
                current_active = active_link.resolve()
            except OSError:
                current_active = None

        # Point previous to current active before switching
        if current_active is not None and current_active.is_dir() and current_active != target_dir.resolve():
            atomic_symlink(current_active, previous_link)

        # Atomically switch active
        atomic_symlink(target_dir, active_link)

        # Record deployment log
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "switch",
            "active_release": release_id,
            "previous_release": previous_link.resolve().name if previous_link.is_symlink() else None,
        }
        with (releases_dir / "history.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "schema_version": 1,
            "status": "activated",
            "active_release": release_id,
            "previous_release": log_entry["previous_release"],
        }


def rollback_release(
    releases_dir: Path = PROJECT_ROOT / "releases",
    project_root: Path = PROJECT_ROOT,
) -> dict:
    with release_lock(releases_dir):
        previous_link = releases_dir / "previous"
        active_link = releases_dir / "active"

        if not previous_link.is_symlink() and not previous_link.exists():
            raise ReleaseError("No previous release reference exists for rollback")

        target_dir = previous_link.resolve()
        if not target_dir.is_dir():
            raise ReleaseError(f"Previous release directory is missing: {target_dir}")

        manifest_file = target_dir / "release_manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        try:
            GRM.validate_manifest(manifest, project_root=project_root)
        except GRM.ReleaseManifestError as exc:
            raise ReleaseError(f"Validation failed for previous release {target_dir.name}: {exc}") from exc

        current_active_id = active_link.resolve().name if active_link.is_symlink() else "unknown"
        target_id = target_dir.name

        # Atomically switch active back to previous
        atomic_symlink(target_dir, active_link)
        previous_link.unlink(missing_ok=True)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "rollback",
            "active_release": target_id,
            "rolled_back_from": current_active_id,
        }
        with (releases_dir / "history.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "schema_version": 1,
            "status": "rolled_back",
            "active_release": target_id,
            "rolled_back_from": current_active_id,
        }


def release_status(releases_dir: Path = PROJECT_ROOT / "releases") -> dict:
    active_link = releases_dir / "active"
    previous_link = releases_dir / "previous"

    active_id = None
    if active_link.is_symlink() or active_link.exists():
        try:
            active_id = active_link.resolve().name
        except OSError:
            active_id = "broken_link"

    previous_id = None
    if previous_link.is_symlink() or previous_link.exists():
        try:
            previous_id = previous_link.resolve().name
        except OSError:
            previous_id = "broken_link"

    registered = []
    if releases_dir.is_dir():
        for child in sorted(releases_dir.iterdir()):
            if child.is_dir() and (child / "release_manifest.json").is_file():
                registered.append(child.name)

    return {
        "schema_version": 1,
        "active_release": active_id,
        "previous_release": previous_id,
        "registered_releases": registered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage FaceGuard release switching and rollback.")
    parser.add_argument(
        "--releases-dir",
        type=Path,
        default=PROJECT_ROOT / "releases",
        help="Releases directory path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reg_parser = subparsers.add_parser("register", help="Register a candidate release")
    reg_parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.yaml",
        help="Path to candidate runtime config",
    )

    switch_parser = subparsers.add_parser("switch", help="Switch active release")
    switch_parser.add_argument("--release-id", required=True, help="Release ID to activate")

    subparsers.add_parser("rollback", help="Roll back to previous release")
    subparsers.add_parser("status", help="Show current release status")

    args = parser.parse_args()

    try:
        if args.command == "register":
            res = register_release(args.config, args.releases_dir, PROJECT_ROOT)
        elif args.command == "switch":
            res = switch_release(args.release_id, args.releases_dir, PROJECT_ROOT)
        elif args.command == "rollback":
            res = rollback_release(args.releases_dir, PROJECT_ROOT)
        elif args.command == "status":
            res = release_status(args.releases_dir)
        else:
            parser.error(f"Unknown command: {args.command}")
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, GRM.ReleaseManifestError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
