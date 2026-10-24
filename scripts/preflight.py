#!/usr/bin/env python3
"""Record environment/package/provider facts without installing anything."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGES = (
    "numpy",
    "Pillow",
    "PyYAML",
    "opencv-python",
    "opencv-python-headless",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "insightface",
    "torch",
    "torchvision",
    "pytest",
)


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout + result.stderr).strip()
    return output or None


def collect_preflight() -> dict[str, Any]:
    packages = {name: _version(name) for name in PACKAGES}
    opencv_flavors = [
        name
        for name in ("opencv-python", "opencv-python-headless")
        if packages[name] is not None
    ]
    errors: list[str] = []
    if len(opencv_flavors) > 1:
        errors.append(f"multiple OpenCV distributions installed: {opencv_flavors}")
    if packages["onnxruntime"] and packages["onnxruntime-gpu"]:
        errors.append("onnxruntime and onnxruntime-gpu are both installed")

    providers: list[str] | None = None
    if packages["onnxruntime"] or packages["onnxruntime-gpu"]:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
        except Exception as exc:  # environment report must record, not hide, import failures
            errors.append(f"onnxruntime import failed: {type(exc).__name__}: {exc}")

    torch_info: dict[str, Any] | None = None
    if packages["torch"]:
        try:
            import torch

            torch_info = {
                "version": torch.__version__,
                "cuda_built": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:
            errors.append(f"torch import failed: {type(exc).__name__}: {exc}")

    pip_check = _command_output([sys.executable, "-m", "pip", "check"])
    if pip_check and "No broken requirements found" not in pip_check:
        errors.append("pip check reported broken or conflicting requirements")

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "onnxruntime_providers": providers,
        "torch": torch_info,
        "nvidia_smi": _command_output(["nvidia-smi"]),
        "pip_check": pip_check,
        "errors": errors,
        "ok": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect_preflight()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, args.output)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
