#!/usr/bin/env python3
"""Run the fail-closed FaceGuard pipeline on a webcam or sequential video."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faceguard.config import ConfigurationError  # noqa: E402
from faceguard.bootstrap import RuntimeFactory, build_readiness_report  # noqa: E402
from faceguard.gallery import GalleryError  # noqa: E402
from faceguard.models import ModelPackError  # noqa: E402
from faceguard.pad import PadInferenceError  # noqa: E402
from faceguard.presentation import operator_message, safe_event  # noqa: E402
from faceguard.runtime import FramePacket, LatestFrameQueue  # noqa: E402
from faceguard.telemetry import MetricsDashboard, MetricsSnapshot, RollingMetrics  # noqa: E402

READ_FAILURE_LIMIT = 3


def build_verifier(config_path: Path):
    bundle = RuntimeFactory.load(config_path, project_root=PROJECT_ROOT)
    return bundle.verifier, bundle.config


def overlay_text(result) -> str:
    """Return the operator message; identity appears only on a PAD-gated verification."""
    return operator_message(result)


def render_and_report(frame, result, sequence: int, no_display: bool, last_state):
    if result.state != last_state:
        print(json.dumps(safe_event(sequence, result), sort_keys=True), flush=True)
        last_state = result.state
    if no_display:
        return last_state, False
    text = overlay_text(result)
    color = (0, 180, 0) if result.status == "verified" else (0, 0, 220)
    cv2.putText(frame, text, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imshow("FaceGuard AI", frame)
    return last_state, cv2.waitKey(1) & 0xFF in (27, ord("q"))


def publish_metrics(
    metrics,
    dashboard,
    result,
    age_ms: float,
    captured: int,
    dropped: int,
    pad_required: int,
) -> None:
    if metrics is None or dashboard is None:
        return
    metrics.add(result.latency_ms, age_ms)
    dashboard.update(
        MetricsSnapshot(
            message=overlay_text(result),
            state=result.state.value,
            status=result.status,
            fps=metrics.fps(),
            median_age_ms=metrics.median_age(),
            p95_age_ms=metrics.p95_age(),
            median_latency_ms=metrics.median_latency(),
            processed=metrics.processed,
            captured=captured,
            dropped=dropped,
            pad_observations=result.pad_observations,
            pad_required=pad_required,
            elapsed_s=metrics.elapsed_s,
        )
    )


def run_video(
    verifier,
    capture,
    no_display: bool,
    metrics=None,
    dashboard=None,
    pad_required: int = 8,
) -> int:
    last_state = None
    sequence = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        sequence += 1
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0.0:
            fps = capture.get(cv2.CAP_PROP_FPS)
            timestamp_ms = sequence * 1000.0 / (fps if fps > 0 else 30.0)
        result = verifier.process(frame, timestamp_ms)
        last_state, stop = render_and_report(frame, result, sequence, no_display, last_state)
        publish_metrics(
            metrics, dashboard, result, result.latency_ms, sequence, 0, pad_required
        )
        if stop or (dashboard is not None and dashboard.closed):
            break
    return 0


def run_camera(
    verifier,
    config,
    capture,
    no_display: bool,
    metrics=None,
    dashboard=None,
    camera_source: int | str | None = None,
    max_reconnect_attempts: int = 3,
    reconnect_delay_s: float = 1.0,
) -> int:
    frames = LatestFrameQueue(maxsize=config.frame_queue_size)
    stopping = threading.Event()
    producer_done = threading.Event()
    read_failed = threading.Event()

    def produce() -> None:
        nonlocal capture
        sequence = 0
        failures = 0
        try:
            while not stopping.is_set():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= READ_FAILURE_LIMIT:
                        if camera_source is None or max_reconnect_attempts <= 0:
                            read_failed.set()
                            break
                        print(
                            json.dumps(
                                {
                                    "event": "camera_disconnected",
                                    "reason": "read_failure_limit_exceeded",
                                    "camera_source": str(camera_source),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if hasattr(verifier, "reset"):
                            verifier.reset()
                        capture.release()

                        reconnected = False
                        for attempt in range(1, max_reconnect_attempts + 1):
                            if stopping.wait(reconnect_delay_s):
                                break
                            print(
                                json.dumps(
                                    {
                                        "attempt": attempt,
                                        "camera_source": str(camera_source),
                                        "event": "camera_reconnecting",
                                        "max_attempts": max_reconnect_attempts,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            new_cap = cv2.VideoCapture(camera_source)
                            if new_cap.isOpened():
                                ok_test, test_frame = new_cap.read()
                                if ok_test and test_frame is not None:
                                    capture = new_cap
                                    reconnected = True
                                    failures = 0
                                    sequence += 1
                                    if hasattr(verifier, "reset"):
                                        verifier.reset()
                                    frames.put_latest(
                                        FramePacket(test_frame, time.monotonic() * 1000.0, sequence)
                                    )
                                    print(
                                        json.dumps(
                                            {
                                                "attempt": attempt,
                                                "camera_source": str(camera_source),
                                                "event": "camera_reconnected",
                                            },
                                            sort_keys=True,
                                        ),
                                        flush=True,
                                    )
                                    break
                                new_cap.release()
                        if not reconnected:
                            read_failed.set()
                            break
                        continue
                    time.sleep(0.05)
                    continue
                failures = 0
                sequence += 1
                frames.put_latest(FramePacket(frame, time.monotonic() * 1000.0, sequence))
        finally:
            producer_done.set()

    producer = threading.Thread(target=produce, name="faceguard-capture", daemon=True)
    producer.start()
    last_state = None
    exit_code = 0
    try:
        while not producer_done.is_set() or frames.size:
            try:
                packet = frames.get(timeout=0.25)
            except queue.Empty:
                if dashboard is not None and dashboard.closed:
                    break
                continue
            try:
                result = verifier.process(packet.frame_bgr, packet.captured_monotonic_ms)
                last_state, stop = render_and_report(
                    packet.frame_bgr, result, packet.sequence, no_display, last_state
                )
                publish_metrics(
                    metrics,
                    dashboard,
                    result,
                    time.monotonic() * 1000.0 - packet.captured_monotonic_ms,
                    packet.sequence,
                    frames.dropped_frames,
                    config.pipeline_policy.pad_min_observations,
                )
                if stop or (dashboard is not None and dashboard.closed):
                    break
            finally:
                frames.task_done()
        if read_failed.is_set():
            print("runtime blocked: camera read failed", file=sys.stderr)
            exit_code = 2
    finally:
        stopping.set()
        capture.release()
        producer.join(timeout=2.0)
        print(
            json.dumps(
                {
                    "event": "capture_summary",
                    "dropped_frames": frames.dropped_frames,
                    "read_failures": read_failed.is_set(),
                    "camera_released": not capture.isOpened(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--metrics-ui",
        action="store_true",
        help="show a separate Tkinter runtime-metrics dashboard",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="load and verify all runtime artifacts without opening a video source",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        help="atomically save the --check-only JSON receipt",
    )
    parser.add_argument(
        "--max-reconnect-attempts",
        type=int,
        default=3,
        help="maximum consecutive camera reconnect attempts on capture loss (default: 3)",
    )
    parser.add_argument(
        "--reconnect-delay-s",
        type=float,
        default=1.0,
        help="seconds to wait between camera reconnect attempts (default: 1.0)",
    )
    args = parser.parse_args(argv)
    if args.check_only and (args.video is not None or args.camera is not None):
        parser.error("--check-only cannot be combined with --video or --camera")
    if not args.check_only and args.video is None and args.camera is None:
        parser.error("one of --video or --camera is required unless --check-only is used")
    if args.readiness_report is not None and not args.check_only:
        parser.error("--readiness-report requires --check-only")
    if args.metrics_ui and args.check_only:
        parser.error("--metrics-ui cannot be combined with --check-only")
    return args


def main() -> int:
    args = parse_args()
    try:
        verifier, config = build_verifier(args.config)
    except (ConfigurationError, GalleryError, ModelPackError, PadInferenceError) as exc:
        print(f"runtime startup blocked: {exc}", file=sys.stderr)
        return 2

    if args.check_only:
        report = build_readiness_report(verifier, config)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        print(rendered, end="")
        if args.readiness_report is not None:
            args.readiness_report.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=args.readiness_report.parent,
                prefix=f".{args.readiness_report.name}.",
                delete=False,
            ) as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            os.replace(temporary, args.readiness_report)
        return 0

    dashboard = None
    metrics = None
    if args.metrics_ui:
        dashboard = MetricsDashboard(
            pad_required=config.pipeline_policy.pad_min_observations
        )
        if dashboard.available:
            metrics = RollingMetrics()
        else:
            print(f"metrics UI unavailable: {dashboard.error}", file=sys.stderr)
            dashboard.close()
            dashboard = None
    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        capture.release()
        if dashboard is not None:
            dashboard.close()
        print("runtime startup blocked: unable to open video source", file=sys.stderr)
        return 2
    interrupted = False
    exit_code = 0
    try:
        if args.video:
            exit_code = run_video(
                verifier,
                capture,
                args.no_display,
                metrics,
                dashboard,
                config.pipeline_policy.pad_min_observations,
            )
        else:
            exit_code = run_camera(
                verifier,
                config,
                capture,
                args.no_display,
                metrics,
                dashboard,
                camera_source=args.camera,
                max_reconnect_attempts=args.max_reconnect_attempts,
                reconnect_delay_s=args.reconnect_delay_s,
            )
    except KeyboardInterrupt:
        interrupted = True
        print(json.dumps({"event": "interrupted", "reason": "keyboard_interrupt"}))
    finally:
        capture.release()
        if dashboard is not None:
            dashboard.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(json.dumps(verifier.end().event(), sort_keys=True))
    return 130 if interrupted else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
