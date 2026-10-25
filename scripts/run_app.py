#!/usr/bin/env python3
"""Launch the centralized local FaceGuard desktop application."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.ui import DesktopApp  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FaceGuard local desktop verification interface."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.yaml",
        help="strict runtime configuration path",
    )
    parser.add_argument(
        "--camera",
        type=int,
        choices=[0],
        default=0,
        help="authorized live camera index (only 0 is supported)",
    )
    parser.add_argument(
        "--identity-labels",
        type=Path,
        default=PROJECT_ROOT / "configs/private/identity_labels.yaml",
        help="optional Git-ignored local display-label configuration",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"desktop startup blocked: {exc}", file=sys.stderr)
        return 2
    try:
        app = DesktopApp(
            root,
            config_path=args.config,
            project_root=PROJECT_ROOT,
            camera_index=args.camera,
            identity_labels_path=args.identity_labels,
        )
        return app.run()
    except (RuntimeError, ValueError) as exc:
        root.destroy()
        print(f"desktop startup blocked: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
