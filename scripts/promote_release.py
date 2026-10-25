#!/usr/bin/env python3
"""Evaluate release candidate eligibility and record explicit promotion sign-off."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from faceguard.config import sha256_file  # noqa: E402
from faceguard.runtime_config import load_runtime_config  # noqa: E402
import generate_release_manifest as GRM  # noqa: E402


class PromotionError(RuntimeError):
    """Raised when release candidate promotion checks fail."""


def run_checks(
    config_path: Path,
    scope: str,
    project_root: Path = PROJECT_ROOT,
    run_pytest: bool = True,
) -> dict[str, Any]:
    check_results: dict[str, Any] = {}

    # 1. Manifest build and validation
    try:
        manifest = GRM.build_manifest(config_path, project_root=project_root)
        check_results["manifest"] = {
            "status": "passed",
            "release_id": manifest["release_id"],
            "artifacts_verified": len(manifest["artifacts"]),
        }
    except Exception as exc:
        raise PromotionError(f"Manifest verification failed: {exc}") from exc

    # 2. Runtime configuration load
    try:
        load_runtime_config(config_path, project_root=project_root)
        check_results["runtime_config"] = {"status": "passed"}
    except Exception as exc:
        raise PromotionError(f"Runtime configuration parsing failed: {exc}") from exc

    # 3. Automated tests
    if run_pytest:
        pytest_cmd = [sys.executable, "-m", "pytest", "-q"]
        res = subprocess.run(
            pytest_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res.returncode != 0:
            raise PromotionError(f"Automated test suite failed:\n{res.stdout}\n{res.stderr}")
        check_results["pytest"] = {"status": "passed"}
    else:
        check_results["pytest"] = {"status": "skipped"}

    # 4. Artifact privacy and credential scanner
    scan_script = project_root / "scripts" / "scan_artifacts.py"
    if scan_script.is_file():
        scan_cmd = [sys.executable, str(scan_script)]
        scan_res = subprocess.run(
            scan_cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if scan_res.returncode != 0:
            raise PromotionError(f"Artifact security scan failed:\n{scan_res.stderr}")
        check_results["artifact_scan"] = {"status": "passed"}

    # 5. Scope-specific gate
    if scope == "production-security":
        raise PromotionError(
            "Promotion to 'production-security' is blocked: PAD generalization error (~20.8%) "
            "and open-set FPIR (1.79%) do not meet production access-control requirements. "
            "Use '--scope academic-prototype' for authorized prototype demonstrations."
        )

    check_results["scope"] = {
        "status": "passed",
        "scope": scope,
        "note": "Authorized for local academic research/demonstration with documented limitations.",
    }

    return check_results


def record_promotion(
    release_id: str,
    scope: str,
    approver: str,
    checks: dict[str, Any],
    output_dir: Path = PROJECT_ROOT / "reports/phase_receipts",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_file = output_dir / f"promotion_{release_id}.json"

    receipt = {
        "schema_version": 1,
        "status": "promoted",
        "release_id": release_id,
        "scope": scope,
        "promoted_by": approver,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "limitations": [
            "Academic prototype demonstration only; not certified for security access control.",
            "PAD model is an RGB single-frame classifier; presentation attack error is ~20.8%.",
            "Open-set recognition exploratory FPIR was 1.79% (5/280 searches).",
        ],
    }

    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{receipt_file.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, receipt_file)
    return receipt_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and promote FaceGuard release candidates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.yaml",
        help="Path to candidate runtime configuration",
    )
    parser.add_argument(
        "--scope",
        choices=["academic-prototype", "production-security"],
        default="academic-prototype",
        help="Target promotion scope",
    )
    parser.add_argument(
        "--approver",
        type=str,
        default=None,
        help="Name of human authority approving promotion",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest run during promotion check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate promotion gates without saving promotion receipt",
    )
    args = parser.parse_args()

    try:
        checks = run_checks(
            args.config,
            scope=args.scope,
            project_root=PROJECT_ROOT,
            run_pytest=not args.skip_tests,
        )
        release_id = checks["manifest"]["release_id"]

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "event": "promotion_evaluation_passed",
                        "status": "validation_passed",
                        "release_id": release_id,
                        "scope": args.scope,
                        "checks": checks,
                    },
                    indent=2,
                )
            )
            return 0

        if not args.approver:
            print(
                json.dumps(
                    {
                        "event": "promotion_awaiting_approval",
                        "status": "awaiting_approval",
                        "release_id": release_id,
                        "scope": args.scope,
                        "checks": checks,
                        "message": "All promotion gates passed. Re-run with --approver <NAME> to complete promotion.",
                    },
                    indent=2,
                )
            )
            return 0

        receipt_path = record_promotion(release_id, args.scope, args.approver, checks)
        print(
            json.dumps(
                {
                    "event": "promotion_completed",
                    "status": "promoted",
                    "release_id": release_id,
                    "scope": args.scope,
                    "promoted_by": args.approver,
                    "receipt_file": str(receipt_path),
                },
                indent=2,
            )
        )
        return 0
    except (PromotionError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
