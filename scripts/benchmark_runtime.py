#!/usr/bin/env python3
"""Record aggregate pipeline performance without retaining frames or embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
import platform
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_demo import READ_FAILURE_LIMIT, build_verifier  # noqa: E402
from faceguard.config import ConfigurationError  # noqa: E402
from faceguard.gallery import GalleryError  # noqa: E402
from faceguard.models import ModelPackError  # noqa: E402
from faceguard.pad import PadInferenceError  # noqa: E402
from faceguard.runtime import FramePacket, LatestFrameQueue  # noqa: E402


def _peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor, 1)


def _ram_gb() -> float | None:
    try:
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1
        )
    except (AttributeError, ValueError, OSError):
        return None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _camera_info(capture: cv2.VideoCapture, index: int) -> dict[str, object]:
    return {
        "index": index,
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "reported_fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 2),
    }


def _summarize(
    latencies: list[float],
    elapsed: float,
    states: Counter,
    reasons: Counter,
) -> dict[str, object]:
    return {
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "processed_fps": round(len(latencies) / elapsed, 2),
        "state_counts": dict(sorted(states.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def run_sequential(verifier, capture, warmup: int, duration: float):
    latencies: list[float] = []
    states: Counter = Counter()
    reasons: Counter = Counter()
    sequence = 0
    verification_started: float | None = None
    verification_completed_ms: float | None = None
    started = time.monotonic()
    while time.monotonic() - started < duration:
        ok, frame = capture.read()
        if not ok:
            break
        sequence += 1
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0.0:
            fps = capture.get(cv2.CAP_PROP_FPS)
            timestamp_ms = sequence * 1000.0 / (fps if fps > 0 else 30.0)
        begin = time.perf_counter()
        result = verifier.process(frame, timestamp_ms)
        observed = time.monotonic()
        if result.state.value == "COLLECTING_PAD" and verification_started is None:
            verification_started = observed
        elif (
            result.state.value == "IDENTIFIED"
            and verification_started is not None
            and verification_completed_ms is None
        ):
            verification_completed_ms = (observed - verification_started) * 1000.0
        if sequence > warmup:
            latencies.append((time.perf_counter() - begin) * 1000.0)
            states[result.state.value] += 1
            reasons[result.reason] += 1
    return (
        latencies,
        states,
        reasons,
        sequence,
        0,
        time.monotonic() - started,
        verification_completed_ms,
    )


def run_live(verifier, config, capture, warmup: int, duration: float):
    frames = LatestFrameQueue(maxsize=config.frame_queue_size)
    stopping = threading.Event()
    done = threading.Event()
    failed = threading.Event()
    captured = [0]

    def produce() -> None:
        failures = 0
        try:
            while not stopping.is_set():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= READ_FAILURE_LIMIT:
                        failed.set()
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                captured[0] += 1
                frames.put_latest(
                    FramePacket(frame, time.monotonic() * 1000.0, captured[0])
                )
        finally:
            done.set()

    producer = threading.Thread(target=produce, name="benchmark-capture", daemon=True)
    producer.start()
    latencies: list[float] = []
    states: Counter = Counter()
    reasons: Counter = Counter()
    verification_started: float | None = None
    verification_completed_ms: float | None = None
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration and not done.is_set():
            try:
                packet = frames.get(timeout=0.25)
            except queue.Empty:
                continue
            result = verifier.process(packet.frame_bgr, packet.captured_monotonic_ms)
            observed = time.monotonic()
            if result.state.value == "COLLECTING_PAD" and verification_started is None:
                verification_started = observed
            elif (
                result.state.value == "IDENTIFIED"
                and verification_started is not None
                and verification_completed_ms is None
            ):
                verification_completed_ms = (observed - verification_started) * 1000.0
            if packet.sequence > warmup:
                latencies.append(time.monotonic() * 1000.0 - packet.captured_monotonic_ms)
                states[result.state.value] += 1
                reasons[result.reason] += 1
            frames.task_done()
    finally:
        stopping.set()
        capture.release()
        producer.join(timeout=2.0)
    return (
        latencies,
        states,
        reasons,
        captured[0],
        frames.dropped_frames,
        failed.is_set(),
        time.monotonic() - started,
        verification_completed_ms,
    )


def _performance_gate(
    session: dict[str, object], objective_path: Path, queue_size: int
) -> dict[str, object]:
    document = yaml.safe_load(objective_path.read_text(encoding="utf-8"))
    if document.get("status") != "FROZEN":
        raise RuntimeError("performance objective must be FROZEN")
    targets = document["objectives"]
    verification_ms = session.get("stable_face_verification_ms")
    peak_memory = session.get("peak_memory_mb")
    checks = {
        "processed_fps": {
            "observed": session["processed_fps"],
            "operator": ">=",
            "target": targets["minimum_processed_fps"],
            "passed": session["processed_fps"] >= targets["minimum_processed_fps"],
        },
        "p95_capture_to_decision_ms": {
            "observed": session["p95_latency_ms"],
            "operator": "<=",
            "target": targets["maximum_p95_capture_to_decision_ms"],
            "passed": session["p95_latency_ms"]
            <= targets["maximum_p95_capture_to_decision_ms"],
        },
        "peak_memory_mb": {
            "observed": peak_memory,
            "operator": "<=",
            "target": targets["maximum_peak_memory_mb"],
            "passed": peak_memory is not None
            and peak_memory <= targets["maximum_peak_memory_mb"],
        },
        "stable_face_verification_seconds": {
            "observed": None
            if verification_ms is None
            else round(float(verification_ms) / 1000.0, 3),
            "operator": "<=",
            "target": targets["maximum_stable_face_verification_seconds"],
            "passed": verification_ms is not None
            and verification_ms / 1000.0
            <= targets["maximum_stable_face_verification_seconds"],
        },
        "read_failures": {
            "observed": int(bool(session["read_failures"])),
            "operator": "<=",
            "target": targets["maximum_read_failures"],
            "passed": int(bool(session["read_failures"]))
            <= targets["maximum_read_failures"],
        },
        "camera_released": {
            "observed": session["camera_released"],
            "operator": "==",
            "target": targets["camera_released_on_shutdown"],
            "passed": session["camera_released"]
            == targets["camera_released_on_shutdown"],
        },
        "frame_queue_size": {
            "observed": queue_size,
            "operator": "==",
            "target": targets["bounded_latest_frame_queue_size"],
            "passed": queue_size == targets["bounded_latest_frame_queue_size"],
        },
    }
    try:
        objective_label = str(objective_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        objective_label = str(objective_path)
    return {
        "objective": objective_label,
        "objective_status": document["status"],
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--video", type=Path)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument(
        "--objective",
        type=Path,
        default=PROJECT_ROOT / "configs/performance_objective.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/benchmarks/prototype_camera.json",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        verifier, config = build_verifier(args.config)
    except (ConfigurationError, GalleryError, ModelPackError, PadInferenceError) as exc:
        print(f"benchmark blocked: {exc}", file=sys.stderr)
        return 2
    if args.seconds <= 0 or args.warmup_frames < 0:
        print("benchmark blocked: seconds must be positive", file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        capture.release()
        print("benchmark blocked: unable to open video source", file=sys.stderr)
        return 2
    camera = _camera_info(capture, args.camera) if args.video is None else None
    try:
        if args.video:
            (
                latencies,
                states,
                reasons,
                captured,
                dropped,
                elapsed,
                verification_completed_ms,
            ) = run_sequential(verifier, capture, args.warmup_frames, args.seconds)
            read_failed = False
        else:
            (
                latencies,
                states,
                reasons,
                captured,
                dropped,
                read_failed,
                elapsed,
                verification_completed_ms,
            ) = run_live(verifier, config, capture, args.warmup_frames, args.seconds)
    finally:
        capture.release()
    elapsed = max(elapsed, 0.001)
    if len(latencies) < 2:
        print("benchmark blocked: insufficient processed frames", file=sys.stderr)
        return 2
    if read_failed:
        print("benchmark blocked: camera read failed", file=sys.stderr)
        return 2

    session = {
        "duration_seconds": round(elapsed, 2),
        "warmup_frames": args.warmup_frames,
        "captured_frames": captured,
        "processed_frames": len(latencies),
        "dropped_frames": dropped,
        "read_failures": read_failed,
        "peak_memory_mb": _peak_memory_mb(),
        "camera_released": not capture.isOpened(),
        "stable_face_verification_ms": None
        if verification_completed_ms is None
        else round(verification_completed_ms, 2),
        **_summarize(latencies, elapsed, states, reasons),
    }
    try:
        performance_gate = _performance_gate(session, args.objective, config.frame_queue_size)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"benchmark blocked: invalid performance objective: {exc}", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": "prototype_benchmark",
        "source": {"type": "video" if args.video else "camera", "camera": camera},
        "target": {
            "os": platform.platform(),
            "cpu": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "ram_gb": _ram_gb(),
            "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "active_provider": config.providers[0],
            "available_providers": list(ort.get_available_providers()),
        },
        "session": session,
        "performance_gate": performance_gate,
        "privacy": {
            "frames_retained": False,
            "embeddings_retained": False,
            "biometric_values_recorded": False,
        },
        "limitations": [
            "Aggregate prototype measurement on one laptop and camera; not a general performance guarantee.",
            "Latency includes capture-to-decision frame age for live camera mode.",
            "Observed state/reason counts show which pipeline stages the session exercised; empty scenes measure capture and detection only.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
