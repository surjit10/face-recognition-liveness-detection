import numpy as np

from faceguard.gallery import GalleryMatch
from faceguard.models import EmbeddingObservation
from faceguard.pipeline import PipelinePolicy, VerificationState, VideoVerifier


class _Analyzer:
    def __init__(self, observations: list[EmbeddingObservation]) -> None:
        self.observations = observations
        self.calls = 0

    def embed_rgb(self, image_rgb: np.ndarray) -> EmbeddingObservation:
        assert image_rgb.shape == (100, 100, 3)
        value = self.observations[min(self.calls, len(self.observations) - 1)]
        self.calls += 1
        return value


class _Pad:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0

    def score_live(self, frame_bgr: np.ndarray, bbox_xyxy) -> float:
        value = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return value


class _Gallery:
    def __init__(self, accepted: bool = True) -> None:
        self.calls = 0
        self.accepted = accepted

    def search(self, embedding, *, threshold, min_margin) -> GalleryMatch:
        self.calls += 1
        return GalleryMatch(
            status="verified" if self.accepted else "unknown",
            person_id="person-a" if self.accepted else None,
            top1_score=0.9,
            top2_score=0.2,
            margin=0.7,
            reason="accepted" if self.accepted else "below_threshold",
        )


def _observation(error: str | None = None) -> EmbeddingObservation:
    if error:
        return EmbeddingObservation(0 if error == "no_face" else 2, None, None, None, error)
    return EmbeddingObservation(
        1,
        np.asarray([1.0, 0.0], dtype=np.float32),
        0.99,
        (20.0, 20.0, 80.0, 80.0),
        None,
    )


def _policy() -> PipelinePolicy:
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


def _frame() -> np.ndarray:
    values = np.indices((100, 100)).sum(axis=0) % 2
    return np.repeat((values * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def test_pad_must_collect_and_pass_before_gallery_is_called() -> None:
    gallery = _Gallery()
    verifier = VideoVerifier(
        analyzer=_Analyzer([_observation()]),
        pad_model=_Pad([0.9, 0.8]),
        gallery=gallery,  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    first = verifier.process(_frame(), 1000.0)
    assert first.state == VerificationState.COLLECTING_PAD
    assert gallery.calls == 0
    second = verifier.process(_frame(), 1100.0)
    assert second.state == VerificationState.IDENTIFIED
    assert second.status == "verified"
    assert second.person_id == "person-a"
    assert gallery.calls == 1


def test_spoof_never_calls_recognition() -> None:
    gallery = _Gallery()
    verifier = VideoVerifier(
        analyzer=_Analyzer([_observation()]),
        pad_model=_Pad([0.1, 0.2]),
        gallery=gallery,  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    verifier.process(_frame(), 1000.0)
    result = verifier.process(_frame(), 1100.0)
    assert result.status == "spoof"
    assert result.state == VerificationState.PAD_FAILED
    assert gallery.calls == 0


def test_unknown_and_reset_paths_are_fail_closed() -> None:
    analyzer = _Analyzer([_observation(), _observation(), _observation("no_face"), _observation()])
    gallery = _Gallery(accepted=False)
    verifier = VideoVerifier(
        analyzer=analyzer,
        pad_model=_Pad([0.9, 0.9, 0.9]),
        gallery=gallery,  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    verifier.process(_frame(), 1000.0)
    unknown = verifier.process(_frame(), 1100.0)
    assert unknown.state == VerificationState.UNKNOWN
    reset = verifier.process(_frame(), 1200.0)
    assert reset.status == "retry"
    assert reset.reason == "no_face"
    new_track = verifier.process(_frame(), 1300.0)
    assert new_track.state == VerificationState.COLLECTING_PAD
    assert new_track.track_id != unknown.track_id


def test_timestamp_regression_resets_track() -> None:
    verifier = VideoVerifier(
        analyzer=_Analyzer([_observation()]),
        pad_model=_Pad([0.9]),
        gallery=_Gallery(),  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    verifier.process(_frame(), 1000.0)
    result = verifier.process(_frame(), 999.0)
    assert result.status == "retry"
    assert result.reason == "timestamp_regression"
    assert verifier.state == VerificationState.RETRY


def test_ended_state_does_not_restart() -> None:
    verifier = VideoVerifier(
        analyzer=_Analyzer([_observation()]),
        pad_model=_Pad([0.9]),
        gallery=_Gallery(),  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    assert verifier.end().state == VerificationState.ENDED
    assert verifier.process(_frame(), 1000.0).status == "ended"


def test_log_safe_event_contains_no_frame_embedding_or_bbox() -> None:
    verifier = VideoVerifier(
        analyzer=_Analyzer([_observation()]),
        pad_model=_Pad([0.9]),
        gallery=_Gallery(),  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )
    event = verifier.process(_frame(), 1000.0).event()
    assert "frame" not in event
    assert "embedding" not in event
    assert "bbox" not in event


def test_multiple_face_boxes_are_available_only_for_in_memory_display() -> None:
    boxes = ((5.0, 5.0, 35.0, 45.0), (55.0, 10.0, 90.0, 50.0))
    observation = EmbeddingObservation(
        2,
        None,
        None,
        None,
        "multiple_faces",
        boxes,
    )
    verifier = VideoVerifier(
        analyzer=_Analyzer([observation]),
        pad_model=_Pad([0.9]),
        gallery=_Gallery(),  # type: ignore[arg-type]
        policy=_policy(),
        model_version="test",
    )

    result = verifier.process(_frame(), 1000.0)

    assert result.reason == "multiple_faces"
    assert result.face_boxes_xyxy == boxes
    event = result.event()
    assert "face_boxes_xyxy" not in event
    assert "bbox" not in event
