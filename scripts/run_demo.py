#!/usr/bin/env python3
"""Run the fail-closed FaceGuard pipeline on a webcam or sequential video."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from faceguard.gallery import Gallery, GalleryError  # noqa: E402
from faceguard.models import InsightFaceEmbedder, ModelPackError  # noqa: E402
from faceguard.pad import OnnxPadModel, PadInferenceError  # noqa: E402
from faceguard.pipeline import VideoVerifier  # noqa: E402
from faceguard.runtime import FramePacket, LatestFrameQueue  # noqa: E402
from faceguard.runtime_config import load_runtime_config  # noqa: E402
from faceguard.telemetry import MetricsDashboard, MetricsSnapshot, RollingMetrics  # noqa: E402

OVERLAY_MESSAGES = {
    "no_face": "NO FACE",
    "multiple_faces": "MULTIPLE FACES",
    "collecting_pad_evidence": "CHECKING LIVENESS",
    "pad_below_threshold": "SPOOF",
}
READ_FAILURE_LIMIT = 3


def build_verifier(config_path: Path):
    config = load_runtime_config(config_path, project_root=PROJECT_ROOT)
    analyzer = InsightFaceEmbedder(
        model_root=config.insightface_model_root,
        pack=config.insightface_pack,
        providers=config.providers,
        expected_hashes={
            config.detector_file: config.detector_sha256,
            config.recognizer_file: config.recognizer_sha256,
        },
    )
    pad = OnnxPadModel(
        config.pad_path,
        expected_sha256=config.pad_sha256,
        providers=config.providers,
        preprocess=config.pad_preprocess,
    )
    gallery = Gallery.load(
        config.gallery_path,
        expected_sha256=config.gallery_sha256,
        expected_embedding_model_sha256=config.recognizer_sha256,
    )
    verifier = VideoVerifier(
        analyzer=analyzer,
        pad_model=pad,
        gallery=gallery,
        policy=config.pipeline_policy,
        model_version=config.release_id,
    )
    return verifier, config


def overlay_text(result) -> str:
    """Return the operator message; identity appears only on a PAD-gated verification."""
    person_id = getattr(result, "person_id", None)
    if result.status == "verified" and person_id:
        return f"VERIFIED: {person_id}"
    if result.status == "spoof":
        return "SPOOF"
    if result.status == "unknown":
        return "UNKNOWN"
    if result.reason in OVERLAY_MESSAGES:
        return OVERLAY_MESSAGES[result.reason]
    return "HOLD STILL"


def render_and_report(frame, result, sequence: int, no_display: bool, last_state):
    if result.state != last_state:
        print(json.dumps({"sequence": sequence, **result.event()}, sort_keys=True), flush=True)
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


def run_camera(verifier, config, capture, no_display: bool, metrics=None, dashboard=None) -> int:
    frames = LatestFrameQueue(maxsize=config.frame_queue_size)
    stopping = threading.Event()
    producer_done = threading.Event()
    read_failed = threading.Event()

    def produce() -> None:
        sequence = 0
        failures = 0
        try:
            while not stopping.is_set():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= READ_FAILURE_LIMIT:
                        read_failed.set()
                        break
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
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event": "runtime_ready",
            "release_id": config.release_id,
            "providers": list(config.providers),
            "face_pack": config.insightface_pack,
            "face_runtime_providers": {
                name: list(providers)
                for name, providers in verifier.analyzer.runtime_providers.items()
            },
            "verified_face_model_hashes": verifier.analyzer.model_hashes,
            "pad_model_sha256": verifier.pad_model.model_sha256,
            "gallery_vectors": verifier.gallery.size,
            "gallery_identities": verifier.gallery.identity_count,
            "camera_opened": False,
        }
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
                verifier, config, capture, args.no_display, metrics, dashboard
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
