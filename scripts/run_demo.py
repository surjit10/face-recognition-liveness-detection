#!/usr/bin/env python3
"""Run the fail-closed FaceGuard pipeline on a webcam or sequential video."""

from __future__ import annotations

import argparse
import json
import queue
import sys
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


def render_and_report(frame, result, sequence: int, no_display: bool, last_state):
    if result.state != last_state or result.status in {"verified", "unknown", "spoof"}:
        event = {"sequence": sequence, **result.event()}
        print(json.dumps(event, sort_keys=True), flush=True)
        last_state = result.state
    stop = False
    if not no_display:
        color = (0, 180, 0) if result.status == "verified" else (0, 0, 220)
        cv2.putText(
            frame,
            f"{result.status.upper()}: {result.reason}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.imshow("FaceGuard AI", frame)
        stop = cv2.waitKey(1) & 0xFF in (27, ord("q"))
    return last_state, stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/runtime.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera", type=int)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    try:
        verifier, config = build_verifier(args.config)
    except (ConfigurationError, GalleryError, ModelPackError, PadInferenceError) as exc:
        parser.exit(2, f"runtime startup blocked: {exc}\n")

    capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not capture.isOpened():
        parser.exit(2, "runtime startup blocked: unable to open video source\n")
    last_state = None
    try:
        if args.video:
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
                last_state, stop = render_and_report(
                    frame, result, sequence, args.no_display, last_state
                )
                if stop:
                    break
        else:
            frames = LatestFrameQueue(maxsize=config.frame_queue_size)
            stopping = threading.Event()
            producer_done = threading.Event()

            def produce() -> None:
                sequence = 0
                try:
                    while not stopping.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            break
                        sequence += 1
                        frames.put_latest(
                            FramePacket(frame, time.monotonic() * 1000.0, sequence)
                        )
                finally:
                    producer_done.set()

            producer = threading.Thread(target=produce, name="faceguard-capture", daemon=True)
            producer.start()
            try:
                while not producer_done.is_set() or frames.size:
                    try:
                        packet = frames.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    try:
                        result = verifier.process(
                            packet.frame_bgr, packet.captured_monotonic_ms
                        )
                        last_state, stop = render_and_report(
                            packet.frame_bgr,
                            result,
                            packet.sequence,
                            args.no_display,
                            last_state,
                        )
                        if stop:
                            break
                    finally:
                        frames.task_done()
            finally:
                stopping.set()
                capture.release()
                producer.join(timeout=2.0)
                print(
                    json.dumps(
                        {"event": "capture_summary", "dropped_frames": frames.dropped_frames}
                    )
                )
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(json.dumps(verifier.end().event(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
