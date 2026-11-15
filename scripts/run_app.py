#!/usr/bin/env python3
"""Launch the centralized local FaceGuard desktop application."""

from __future__ import annotations

import argparse
import signal
import sys
import tkinter as tk
import warnings
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=r"`estimate` is deprecated.*",
    category=FutureWarning,
    module=r"insightface\.utils\.face_align",
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.live_metrics import (  # noqa: E402
    LiveMetricsHub,
    LiveMetricsServer,
    load_performance_targets,
)
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
    parser.add_argument(
        "--history-db",
        type=Path,
        default=PROJECT_ROOT / "reports/private/checkins.sqlite3",
        help="private local SQLite check-in history path",
    )
    parser.add_argument(
        "--operator-config",
        type=Path,
        default=PROJECT_ROOT / "configs/private/operator.yaml",
        help="optional Git-ignored operator PIN configuration",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        help="serve live aggregate metrics and the MLOps dashboard on "
        "http://127.0.0.1:PORT/ (loopback only; e.g. 8765)",
    )
    args = parser.parse_args(argv)
    if args.metrics_port is not None and not 1024 <= args.metrics_port <= 65535:
        parser.error("--metrics-port must be within 1024-65535")
    return args


def start_live_metrics(port: int | None):
    """Start the opt-in loopback metrics server; failure never blocks the desktop app."""
    if port is None:
        return None, None
    try:
        targets = load_performance_targets(PROJECT_ROOT / "configs/performance_objective.yaml")
    except (OSError, ValueError, AttributeError):
        targets = {}
    hub = LiveMetricsHub(source="desktop", targets=targets)
    try:
        server = LiveMetricsServer(
            hub, port=port, dashboard=PROJECT_ROOT / "reports/mlops_dashboard.html"
        ).start()
    except (OSError, ValueError) as exc:
        print(f"live metrics unavailable: {exc}", file=sys.stderr)
        return None, None
    print(f"live metrics: {server.url}", flush=True)
    return hub, server


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"desktop startup blocked: {exc}", file=sys.stderr)
        return 2
    hub, server = start_live_metrics(args.metrics_port)
    try:
        app = DesktopApp(
            root,
            config_path=args.config,
            project_root=PROJECT_ROOT,
            camera_index=args.camera,
            identity_labels_path=args.identity_labels,
            history_db_path=args.history_db,
            operator_config_path=args.operator_config,
            metrics_hub=hub,
        )
        signal.signal(signal.SIGINT, lambda *_args: root.after(0, app.close))
        return app.run()
    except (RuntimeError, ValueError) as exc:
        root.destroy()
        print(f"desktop startup blocked: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
