#!/usr/bin/env python3
"""Fail when tracked files contain restricted artifacts or credential-shaped text."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".mp4",
    ".zip",
    ".npy",
    ".npz",
    ".onnx",
    ".pt",
    ".pth",
    ".pem",
    ".key",
}
FORBIDDEN_PREFIXES = (
    "models/private/",
    "data/raw/",
    "data/processed/",
    "reports/private/",
)
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*"
        r"['\"][^'\"\s]{8,}['\"]"
    ),
)


def scan(root: Path, paths: Iterable[str]) -> list[str]:
    findings: list[str] = []
    for relative in sorted(set(paths)):
        if not relative:
            continue
        path = root / relative
        suffix = path.suffix.lower()
        name = path.name.lower()
        placeholder = name == ".gitkeep"
        if not placeholder and (
            suffix in FORBIDDEN_SUFFIXES
            or relative.startswith(FORBIDDEN_PREFIXES)
            or name == ".env"
            or name.startswith(("credentials", "secrets"))
        ):
            findings.append(f"restricted path: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for pattern in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(f"credential-shaped value: {relative}:{line_number}")
                    break
    return findings


def _git(root: Path, *arguments: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [value for value in result.stdout.split("\0") if value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        paths = _git(args.root, "ls-files", "-z")
        paths += _git(args.root, "diff", "--cached", "--name-only", "-z")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"artifact scan blocked: {exc}", file=sys.stderr)
        return 2
    findings = scan(args.root, paths)
    for finding in findings:
        print(finding)
    if findings:
        print(f"artifact scan failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"artifact scan passed: {len(set(paths))} tracked/staged path(s) reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
