import numpy as np
import pytest

from faceguard.recognition_metrics import (
    RecognitionMetricError,
    evaluate_recognition_operating_point,
    select_recognition_threshold,
)


def test_minimum_balanced_error_threshold_is_deterministic() -> None:
    report = select_recognition_threshold(
        np.asarray([0.9, 0.8, 0.7]),
        np.asarray([0.4, 0.3, 0.2]),
        policy="min_recognition_balanced_error",
    )
    assert report["selected"]["fmr"] == 0.0
    assert report["selected"]["fnmr"] == 0.0
    assert 0.4 < report["threshold"] <= 0.7


def test_fmr_bound_policy_and_validation() -> None:
    report = select_recognition_threshold(
        np.asarray([0.8, 0.6]),
        np.asarray([0.7, 0.2]),
        policy="recognition_fmr_bound",
        max_fmr=0.0,
    )
    assert report["selected"]["fmr"] == 0.0
    with pytest.raises(RecognitionMetricError):
        select_recognition_threshold(
            np.asarray([0.8]),
            np.asarray([0.2]),
            policy="recognition_fmr_bound",
        )


def test_wrong_identity_and_ambiguous_margin_count_as_non_matches() -> None:
    report = select_recognition_threshold(
        np.asarray([0.95, 0.90]),
        np.asarray([0.20, 0.10]),
        policy="min_recognition_balanced_error",
        known_correct=np.asarray([True, False]),
        known_margins=np.asarray([0.3, 0.3]),
        impostor_margins=np.asarray([0.3, 0.01]),
        min_margin=0.1,
    )
    assert report["selected"]["false_non_matches"] == 1
    assert report["selected"]["false_matches"] == 0


def test_frozen_operating_point_separates_rejection_and_misidentification() -> None:
    report = evaluate_recognition_operating_point(
        np.asarray([0.9, 0.8, 0.3]),
        np.asarray([0.7, 0.2]),
        threshold=0.6,
        min_margin=0.1,
        known_correct=np.asarray([True, False, True]),
        known_margins=np.asarray([0.2, 0.2, 0.2]),
        impostor_margins=np.asarray([0.05, 0.2]),
    )
    assert report["known_correct_accepts"] == 1
    assert report["known_misidentification_accepts"] == 1
    assert report["known_rejections"] == 1
    assert report["impostor_false_accepts"] == 0
