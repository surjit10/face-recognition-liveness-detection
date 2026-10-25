"""Fail-closed temporal PAD-before-recognition verification state machine."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np

from .gallery import Gallery, GalleryError, GalleryMatch
from .models import EmbeddingObservation
from .pad import PadInferenceError


class VerificationState(str, Enum):
    EMPTY = "EMPTY"
    COLLECTING_PAD = "COLLECTING_PAD"
    PAD_FAILED = "PAD_FAILED"
    PAD_PASSED = "PAD_PASSED"
    IDENTIFIED = "IDENTIFIED"
    UNKNOWN = "UNKNOWN"
    RETRY = "RETRY"
    ENDED = "ENDED"


class FaceAnalyzer(Protocol):
    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation: ...


class PadScorer(Protocol):
    def score_live(
        self, frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]
    ) -> float: ...


@dataclass(frozen=True)
class PipelinePolicy:
    pad_threshold: float
    recognition_threshold: float
    recognition_min_margin: float
    pad_min_observations: int
    max_track_gap_ms: float
    min_iou: float
    min_detection_score: float
    min_face_size_px: int
    min_blur_variance: float
    min_mean_brightness: float
    max_mean_brightness: float

    def validate(self) -> None:
        bounded = {
            "pad_threshold": self.pad_threshold,
            "recognition_threshold": self.recognition_threshold,
            "min_iou": self.min_iou,
            "min_detection_score": self.min_detection_score,
        }
        for name, value in bounded.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not 0.0 <= self.recognition_min_margin <= 2.0:
            raise ValueError("recognition_min_margin must be within [0, 2]")
        if self.pad_min_observations <= 0 or self.min_face_size_px <= 0:
            raise ValueError("Observation count and minimum face size must be positive")
        if self.max_track_gap_ms <= 0 or self.min_blur_variance < 0:
            raise ValueError("Track gap must be positive and blur threshold non-negative")
        if not 0 <= self.min_mean_brightness < self.max_mean_brightness <= 255:
            raise ValueError("Brightness range must satisfy 0 <= min < max <= 255")


@dataclass(frozen=True)
class VerificationResult:
    status: str
    state: VerificationState
    person_id: str | None
    similarity: float | None
    pad_score: float | None
    track_id: int | None
    model_version: str
    reason: str
    latency_ms: float
    pad_observations: int = 0

    def event(self) -> dict[str, object]:
        """Return a log-safe event; frames and embeddings are intentionally absent."""
        return {
            "status": self.status,
            "state": self.state.value,
            "person_id": self.person_id,
            "similarity": self.similarity,
            "pad_score": self.pad_score,
            "track_id": self.track_id,
            "model_version": self.model_version,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "pad_observations": self.pad_observations,
        }


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


class VideoVerifier:
    """Maintain evidence for exactly one short-lived face track and one session."""

    def __init__(
        self,
        *,
        analyzer: FaceAnalyzer,
        pad_model: PadScorer,
        gallery: Gallery,
        policy: PipelinePolicy,
        model_version: str,
    ) -> None:
        policy.validate()
        self.analyzer = analyzer
        self.pad_model = pad_model
        self.gallery = gallery
        self.policy = policy
        self.model_version = model_version
        self._next_track_id = 1
        self.reset()

    @property
    def state(self) -> VerificationState:
        return self._state

    def reset(self) -> None:
        self._state = VerificationState.EMPTY
        self._track_id: int | None = None
        self._bbox: tuple[float, float, float, float] | None = None
        self._last_timestamp_ms: float | None = None
        self._pad_scores: list[float] = []
        self._terminal_result: VerificationResult | None = None

    def _result(
        self,
        started: float,
        *,
        status: str,
        state: VerificationState,
        reason: str,
        pad_score: float | None = None,
        match: GalleryMatch | None = None,
    ) -> VerificationResult:
        self._state = state
        return VerificationResult(
            status=status,
            state=state,
            person_id=None if match is None else match.person_id,
            similarity=None if match is None else match.top1_score,
            pad_score=pad_score,
            track_id=self._track_id,
            model_version=self.model_version,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            pad_observations=len(self._pad_scores),
        )

    def _retry_and_reset(self, started: float, reason: str) -> VerificationResult:
        old_track = self._track_id
        self.reset()
        self._state = VerificationState.RETRY
        return VerificationResult(
            status="retry",
            state=VerificationState.RETRY,
            person_id=None,
            similarity=None,
            pad_score=None,
            track_id=old_track,
            model_version=self.model_version,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            pad_observations=len(self._pad_scores),
        )

    def _quality_reason(
        self,
        frame_bgr: np.ndarray,
        observation: EmbeddingObservation,
    ) -> str | None:
        if observation.detection_score is None or (
            observation.detection_score < self.policy.min_detection_score
        ):
            return "low_detection_score"
        if observation.bbox_xyxy is None:
            return "missing_face_box"
        x1, y1, x2, y2 = observation.bbox_xyxy
        if min(x2 - x1, y2 - y1) < self.policy.min_face_size_px:
            return "face_too_small"
        height, width = frame_bgr.shape[:2]
        ix1, iy1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        ix2, iy2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
        crop = frame_bgr[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return "empty_face_crop"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if not self.policy.min_mean_brightness <= brightness <= self.policy.max_mean_brightness:
            return "exposure_out_of_range"
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur < self.policy.min_blur_variance:
            return "face_too_blurry"
        return None

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> VerificationResult:
        started = time.perf_counter()
        if self._state == VerificationState.ENDED:
            return self._result(
                started, status="ended", state=VerificationState.ENDED, reason="stream_ended"
            )
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.dtype != np.uint8
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
        ):
            return self._retry_and_reset(started, "invalid_frame")
        if not np.isfinite(timestamp_ms):
            return self._retry_and_reset(started, "invalid_timestamp")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            return self._retry_and_reset(started, "timestamp_regression")

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            observation = self.analyzer.embed_rgb(rgb)
        except Exception:
            return self._retry_and_reset(started, "face_model_error")
        if observation.error is not None:
            return self._retry_and_reset(started, observation.error)
        quality_reason = self._quality_reason(frame_bgr, observation)
        if quality_reason is not None:
            return self._retry_and_reset(started, quality_reason)
        assert observation.bbox_xyxy is not None
        assert observation.embedding is not None

        continuing = (
            self._bbox is not None
            and self._last_timestamp_ms is not None
            and timestamp_ms - self._last_timestamp_ms <= self.policy.max_track_gap_ms
            and bbox_iou(self._bbox, observation.bbox_xyxy) >= self.policy.min_iou
        )
        if not continuing:
            self.reset()
            self._track_id = self._next_track_id
            self._next_track_id += 1
            self._state = VerificationState.COLLECTING_PAD
        self._bbox = observation.bbox_xyxy
        self._last_timestamp_ms = float(timestamp_ms)

        if continuing and self._terminal_result is not None:
            cached = self._terminal_result
            return VerificationResult(
                status=cached.status,
                state=cached.state,
                person_id=cached.person_id,
                similarity=cached.similarity,
                pad_score=cached.pad_score,
                track_id=self._track_id,
                model_version=self.model_version,
                reason="cached_continuous_track",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                pad_observations=cached.pad_observations,
            )

        try:
            pad_score = float(self.pad_model.score_live(frame_bgr, observation.bbox_xyxy))
        except (PadInferenceError, ValueError, RuntimeError):
            return self._retry_and_reset(started, "pad_model_error")
        if not np.isfinite(pad_score) or not 0.0 <= pad_score <= 1.0:
            return self._retry_and_reset(started, "invalid_pad_score")
        self._pad_scores.append(pad_score)
        aggregate = float(np.median(self._pad_scores))
        if len(self._pad_scores) < self.policy.pad_min_observations:
            return self._result(
                started,
                status="retry",
                state=VerificationState.COLLECTING_PAD,
                reason="collecting_pad_evidence",
                pad_score=aggregate,
            )
        if aggregate < self.policy.pad_threshold:
            result = self._result(
                started,
                status="spoof",
                state=VerificationState.PAD_FAILED,
                reason="pad_below_threshold",
                pad_score=aggregate,
            )
            self._terminal_result = result
            return result

        self._state = VerificationState.PAD_PASSED
        try:
            match = self.gallery.search(
                observation.embedding,
                threshold=self.policy.recognition_threshold,
                min_margin=self.policy.recognition_min_margin,
            )
        except GalleryError:
            return self._retry_and_reset(started, "gallery_error")
        if match.status == "verified":
            result = self._result(
                started,
                status="verified",
                state=VerificationState.IDENTIFIED,
                reason="pad_passed_identity_accepted",
                pad_score=aggregate,
                match=match,
            )
        else:
            result = self._result(
                started,
                status="unknown",
                state=VerificationState.UNKNOWN,
                reason=f"pad_passed_identity_{match.reason}",
                pad_score=aggregate,
                match=match,
            )
        self._terminal_result = result
        return result

    def end(self) -> VerificationResult:
        started = time.perf_counter()
        self.reset()
        return self._result(
            started, status="ended", state=VerificationState.ENDED, reason="stream_ended"
        )
