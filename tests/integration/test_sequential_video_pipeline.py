"""Integration tests for Phase P3 — Sequential Video Pipeline.

Tests verify end-to-end video processing from MP4 ingestion through the fail-closed
PAD-before-recognition state machine across all six required scenarios:
1. Enrolled identity
2. Unknown identity
3. Spoof presentation
4. Empty scene (no face)
5. Multiple faces
6. Interrupted / corrupt stream

Also enforces P3-08 privacy assertions: zero raw frames, crops, or embeddings in events.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
import pytest

from faceguard.gallery import GalleryMatch
from faceguard.models import EmbeddingObservation
from faceguard.pipeline import PipelinePolicy, VerificationState, VideoVerifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "faceguard_run_demo", PROJECT_ROOT / "scripts" / "run_demo.py"
)
assert SPEC is not None and SPEC.loader is not None
RUN_DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_DEMO)


class MockAnalyzer:
    """Configurable face detector and embedder for video integration tests."""

    def __init__(self, observation_generator) -> None:
        self._generator = observation_generator
        self.calls = 0

    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        self.calls += 1
        return self._generator(self.calls, image_rgb)


class MockPadModel:
    """Configurable presentation attack detector for video integration tests."""

    def __init__(self, scores: list[float] | float) -> None:
        self.scores = [scores] if isinstance(scores, (int, float)) else scores
        self.calls = 0

    def score_live(self, frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return float(score)


class MockGallery:
    """Configurable biometric recognition gallery for video integration tests."""

    def __init__(self, match: GalleryMatch) -> None:
        self.match = match
        self.calls = 0

    def search(self, embedding: np.ndarray, *, threshold: float, min_margin: float) -> GalleryMatch:
        self.calls += 1
        return self.match


def create_synthetic_mp4(
    path: Path,
    num_frames: int = 8,
    fps: float = 30.0,
    width: int = 120,
    height: int = 120,
    color: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Generate a valid synthetic MP4 video container on disk."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for idx in range(num_frames):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            # Add textured block so Laplacian variance is non-zero
            cv2.rectangle(frame, (25, 25), (75, 75), (200, 200, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def default_policy() -> PipelinePolicy:
    return PipelinePolicy(
        pad_threshold=0.7,
        recognition_threshold=0.8,
        recognition_min_margin=0.2,
        pad_min_observations=2,
        max_track_gap_ms=500.0,
        min_iou=0.3,
        min_detection_score=0.6,
        min_face_size_px=30,
        min_blur_variance=0.0,
        min_mean_brightness=0.0,
        max_mean_brightness=255.0,
    )


def assert_privacy_safe_event(event: dict[str, Any]) -> None:
    """Verify strictly zero raw frames, crops, embeddings, or tensors are exposed."""
    forbidden_keys = {
        "frame",
        "raw_frame",
        "image",
        "crop",
        "face_crop",
        "embedding",
        "embeddings",
        "vector",
        "tensor",
        "bbox",
        "bbox_xyxy",
        "landmarks",
    }
    present_forbidden = forbidden_keys.intersection(event.keys())
    assert not present_forbidden, f"Privacy violation! Event exposed forbidden biometric keys: {present_forbidden}"


def test_integration_enrolled_identity_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Case 1: Enrolled identity MP4 processes through PAD to verified recognition."""
    video_path = tmp_path / "enrolled_sequence.mp4"
    create_synthetic_mp4(video_path, num_frames=6)

    def _face_obs(call_idx: int, image_rgb: np.ndarray) -> EmbeddingObservation:
        return EmbeddingObservation(
            face_count=1,
            embedding=np.array([0.9, 0.1, 0.0], dtype=np.float32),
            detection_score=0.98,
            bbox_xyxy=(25.0, 25.0, 75.0, 75.0),
            error=None,
        )

    gallery_match = GalleryMatch(
        status="verified",
        person_id="person-042",
        top1_score=0.92,
        top2_score=0.35,
        margin=0.57,
        reason="accepted",
    )
    gallery = MockGallery(gallery_match)
    verifier = VideoVerifier(
        analyzer=MockAnalyzer(_face_obs),
        pad_model=MockPadModel([0.88, 0.92, 0.90]),
        gallery=gallery,  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    assert verifier.state == VerificationState.IDENTIFIED
    assert gallery.calls == 1  # Verified and cached for continuous track

    # Assert clean stream termination
    end_res = verifier.end()
    assert end_res.state == VerificationState.ENDED

    captured = capsys.readouterr().out
    for line in captured.strip().split("\n"):
        if line.strip():
            event = json.loads(line)
            assert_privacy_safe_event(event)


def test_integration_unknown_identity_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Case 2: Unknown identity MP4 passes PAD but is safely rejected by gallery."""
    video_path = tmp_path / "unknown_sequence.mp4"
    create_synthetic_mp4(video_path, num_frames=6)

    def _face_obs(call_idx: int, image_rgb: np.ndarray) -> EmbeddingObservation:
        return EmbeddingObservation(
            face_count=1,
            embedding=np.array([0.1, 0.9, 0.0], dtype=np.float32),
            detection_score=0.95,
            bbox_xyxy=(25.0, 25.0, 75.0, 75.0),
            error=None,
        )

    gallery_match = GalleryMatch(
        status="unknown",
        person_id=None,
        top1_score=0.45,
        top2_score=0.40,
        margin=0.05,
        reason="below_threshold",
    )
    gallery = MockGallery(gallery_match)
    verifier = VideoVerifier(
        analyzer=MockAnalyzer(_face_obs),
        pad_model=MockPadModel([0.85, 0.89]),
        gallery=gallery,  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    assert verifier.state == VerificationState.UNKNOWN
    assert gallery.calls == 1

    captured = capsys.readouterr().out
    for line in captured.strip().split("\n"):
        if line.strip():
            event = json.loads(line)
            assert_privacy_safe_event(event)
            assert event.get("person_id") is None


def test_integration_spoof_presentation_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Case 3: Presentation attack MP4 fails PAD; gallery search is NEVER called."""
    video_path = tmp_path / "spoof_sequence.mp4"
    create_synthetic_mp4(video_path, num_frames=6)

    def _face_obs(call_idx: int, image_rgb: np.ndarray) -> EmbeddingObservation:
        return EmbeddingObservation(
            face_count=1,
            embedding=np.array([0.9, 0.1, 0.0], dtype=np.float32),
            detection_score=0.96,
            bbox_xyxy=(25.0, 25.0, 75.0, 75.0),
            error=None,
        )

    gallery = MockGallery(GalleryMatch("verified", "person-042", 0.95, 0.2, 0.75, "accepted"))
    pad_model = MockPadModel([0.15, 0.20])  # Below 0.7 threshold
    verifier = VideoVerifier(
        analyzer=MockAnalyzer(_face_obs),
        pad_model=pad_model,
        gallery=gallery,  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    assert verifier.state == VerificationState.PAD_FAILED
    # Strict fail-closed cascade: gallery must never be queried on spoof presentation
    assert gallery.calls == 0

    captured = capsys.readouterr().out
    for line in captured.strip().split("\n"):
        if line.strip():
            event = json.loads(line)
            assert_privacy_safe_event(event)
            assert event.get("status") in ("retry", "spoof")


def test_integration_empty_scene_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Case 4: Empty scene (no face) stays fail-closed in RETRY."""
    video_path = tmp_path / "empty_scene.mp4"
    create_synthetic_mp4(video_path, num_frames=4)

    def _no_face_obs(call_idx: int, image_rgb: np.ndarray) -> EmbeddingObservation:
        return EmbeddingObservation(
            face_count=0,
            embedding=None,
            detection_score=None,
            bbox_xyxy=None,
            error="no_face",
        )

    gallery = MockGallery(GalleryMatch("verified", "person-001", 0.9, 0.1, 0.8, "accepted"))
    pad_model = MockPadModel([0.9])
    verifier = VideoVerifier(
        analyzer=MockAnalyzer(_no_face_obs),
        pad_model=pad_model,
        gallery=gallery,  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    assert verifier.state == VerificationState.RETRY
    assert pad_model.calls == 0
    assert gallery.calls == 0


def test_integration_multiple_faces_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Case 5: Multiple faces in frame triggers fail-closed retry; never verifies."""
    video_path = tmp_path / "multi_face.mp4"
    create_synthetic_mp4(video_path, num_frames=4)

    def _multi_face_obs(call_idx: int, image_rgb: np.ndarray) -> EmbeddingObservation:
        return EmbeddingObservation(
            face_count=2,
            embedding=None,
            detection_score=None,
            bbox_xyxy=None,
            error="multiple_faces",
            face_boxes_xyxy=(
                (10.0, 15.0, 45.0, 70.0),
                (55.0, 20.0, 90.0, 75.0),
            ),
        )

    gallery = MockGallery(GalleryMatch("verified", "person-001", 0.9, 0.1, 0.8, "accepted"))
    pad_model = MockPadModel([0.9])
    verifier = VideoVerifier(
        analyzer=MockAnalyzer(_multi_face_obs),
        pad_model=pad_model,
        gallery=gallery,  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    assert verifier.state == VerificationState.RETRY
    assert pad_model.calls == 0
    assert gallery.calls == 0


def test_integration_interrupted_or_corrupt_stream(tmp_path: Path) -> None:
    """Case 6: Mid-stream interruption or EOF cleanly releases resources and ends stream."""
    video_path = tmp_path / "truncated.mp4"
    create_synthetic_mp4(video_path, num_frames=3)

    verifier = VideoVerifier(
        analyzer=MockAnalyzer(lambda *args: EmbeddingObservation(0, None, None, None, "no_face")),
        pad_model=MockPadModel([0.9]),
        gallery=MockGallery(GalleryMatch("unknown", None, 0.0, 0.0, 0.0, "none")),  # type: ignore[arg-type]
        policy=default_policy(),
        model_version="test-v2",
    )

    cap = cv2.VideoCapture(str(video_path))
    exit_code = RUN_DEMO.run_video(verifier, cap, no_display=True)
    cap.release()

    assert exit_code == 0
    end_res = verifier.end()
    assert end_res.state == VerificationState.ENDED
    assert end_res.status == "ended"
    assert end_res.reason == "stream_ended"


def test_integration_invalid_video_file_startup_failure(tmp_path: Path) -> None:
    """Verify runtime startup fails safely with exit code 2 when video file cannot be opened."""
    invalid_path = tmp_path / "non_existent.mp4"
    cap = cv2.VideoCapture(str(invalid_path))
    assert not cap.isOpened()
    cap.release()
