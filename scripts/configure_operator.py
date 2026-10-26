#!/usr/bin/env python3
"""Create a private PBKDF2 operator PIN configuration without storing plaintext."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import ConfigurationError  # noqa: E402
from faceguard.operator_auth import build_operator_document  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Configure the local FaceGuard operator PIN.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs/private/operator.yaml",
        help="Git-ignored private operator configuration path",
    )
    parser.add_argument(
        "--role",
        choices=("attendance-admin", "operator"),
        default="attendance-admin",
        help="attendance-admin can manage attendance only; operator can access all screens",
    )
    args = parser.parse_args(argv)
    role_label = "attendance admin" if args.role == "attendance-admin" else "operator"
    first = getpass.getpass(f"Choose {role_label} PIN (4-64 characters): ")
    second = getpass.getpass(f"Confirm {role_label} PIN: ")
    if first != second:
        print(f"{args.role} setup blocked: PIN values do not match", file=sys.stderr)
        return 2
    try:
        document = build_operator_document(first, role=args.role.replace("-", "_"))
        rendered = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            delete=False,
        ) as handle:
            handle.write(rendered)
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, args.output)
    except (ConfigurationError, OSError) as exc:
        print(f"operator setup blocked: {exc}", file=sys.stderr)
        return 2
    print(f"{args.role} PIN configured: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
