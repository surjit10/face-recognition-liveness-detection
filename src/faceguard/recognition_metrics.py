"""Deterministic validation-only recognition threshold selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class RecognitionMetricError(RuntimeError):
    """Recognition calibration inputs or policy are invalid."""


def evaluate_recognition_operating_point(
    known_scores: np.ndarray,
    impostor_scores: np.ndarray,
    *,
    threshold: float,
    min_margin: float,
    known_correct: np.ndarray,
    known_margins: np.ndarray,
    impostor_margins: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a frozen open-set threshold/margin without selecting either value."""
    known = _scores(known_scores, "Known-person")
    impostor = _scores(impostor_scores, "Impostor")
    correct = np.asarray(known_correct, dtype=bool)
    known_margin_values = np.asarray(known_margins, dtype=np.float64)
    impostor_margin_values = np.asarray(impostor_margins, dtype=np.float64)
    if correct.shape != known.shape:
        raise RecognitionMetricError("Known correctness flags must align with scores")
    if known_margin_values.shape != known.shape or impostor_margin_values.shape != impostor.shape:
        raise RecognitionMetricError("Margins must align with recognition scores")
    if not np.isfinite(np.concatenate((known_margin_values, impostor_margin_values))).all():
        raise RecognitionMetricError("Recognition margins must be finite")
    if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise RecognitionMetricError("Recognition threshold must be within [-1, 1]")
    if not np.isfinite(min_margin) or not 0.0 <= min_margin <= 2.0:
        raise RecognitionMetricError("Recognition margin must be within [0, 2]")
    known_accept = (known >= threshold) & (known_margin_values >= min_margin)
    impostor_accept = (impostor >= threshold) & (impostor_margin_values >= min_margin)
    correct_accept = known_accept & correct
    misidentified_accept = known_accept & ~correct
    return {
        "threshold": threshold,
        "min_top1_top2_margin": min_margin,
        "known_searches": len(known),
        "known_correct_accepts": int(np.sum(correct_accept)),
        "known_misidentification_accepts": int(np.sum(misidentified_accept)),
        "known_rejections": int(np.sum(~known_accept)),
        "known_identification_failure_rate": float(1.0 - np.mean(correct_accept)),
        "known_misidentification_accept_rate": float(np.mean(misidentified_accept)),
        "impostor_searches": len(impostor),
        "impostor_false_accepts": int(np.sum(impostor_accept)),
        "impostor_false_positive_identification_rate": float(np.mean(impostor_accept)),
    }


def _scores(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise RecognitionMetricError(f"{name} scores must be finite and one-dimensional")
    if np.any(result < -1.0) or np.any(result > 1.0):
        raise RecognitionMetricError(f"{name} cosine scores must be within [-1, 1]")
    return result


def select_recognition_threshold(
    known_scores: np.ndarray,
    impostor_scores: np.ndarray,
    *,
    policy: str,
    max_fmr: float | None = None,
    known_correct: np.ndarray | None = None,
    known_margins: np.ndarray | None = None,
    impostor_margins: np.ndarray | None = None,
    min_margin: float = 0.0,
) -> dict[str, Any]:
    """Select a threshold from validation search scores using runtime's >= rule."""
    known = _scores(known_scores, "Known-person")
    impostor = _scores(impostor_scores, "Impostor")
    correct = (
        np.ones(len(known), dtype=bool)
        if known_correct is None
        else np.asarray(known_correct, dtype=bool)
    )
    known_margin_values = (
        np.full(len(known), 2.0)
        if known_margins is None
        else np.asarray(known_margins, dtype=np.float64)
    )
    impostor_margin_values = (
        np.full(len(impostor), 2.0)
        if impostor_margins is None
        else np.asarray(impostor_margins, dtype=np.float64)
    )
    if correct.shape != known.shape:
        raise RecognitionMetricError("Known correctness flags must align with known scores")
    if (
        known_margin_values.shape != known.shape
        or impostor_margin_values.shape != impostor.shape
        or not np.isfinite(np.concatenate((known_margin_values, impostor_margin_values))).all()
    ):
        raise RecognitionMetricError("Recognition margins must be finite and align with scores")
    if not np.isfinite(min_margin) or min_margin < 0.0:
        raise RecognitionMetricError("Minimum recognition margin must be finite and non-negative")
    unique = np.unique(np.concatenate((known, impostor)))
    thresholds = np.concatenate(
        ([np.nextafter(unique[-1], math.inf)], unique[::-1], [np.nextafter(unique[0], -math.inf)])
    )
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        known_accept = correct & (known_margin_values >= min_margin) & (known >= threshold)
        impostor_accept = (impostor_margin_values >= min_margin) & (impostor >= threshold)
        false_non_matches = int(np.sum(~known_accept))
        false_matches = int(np.sum(impostor_accept))
        fnmr = false_non_matches / len(known)
        fmr = false_matches / len(impostor)
        curve.append(
            {
                "threshold": float(threshold),
                "false_matches": false_matches,
                "true_rejects": int(len(impostor) - false_matches),
                "false_non_matches": false_non_matches,
                "true_matches": int(len(known) - false_non_matches),
                "fmr": fmr,
                "fnmr": fnmr,
                "balanced_error": (fmr + fnmr) / 2.0,
            }
        )
    if policy == "min_recognition_balanced_error":
        chosen = min(curve, key=lambda row: (row["balanced_error"], -row["threshold"]))
    elif policy == "recognition_fmr_bound":
        if max_fmr is None or not np.isfinite(max_fmr) or not 0.0 <= max_fmr <= 1.0:
            raise RecognitionMetricError("recognition_fmr_bound requires max_fmr in [0, 1]")
        feasible = [row for row in curve if row["fmr"] <= max_fmr]
        if not feasible:
            raise RecognitionMetricError("No recognition threshold satisfies the FMR bound")
        chosen = min(feasible, key=lambda row: (row["fnmr"], -row["threshold"]))
    else:
        raise RecognitionMetricError(f"Unsupported recognition threshold policy: {policy}")
    return {
        "policy": policy,
        "max_fmr": max_fmr,
        "min_top1_top2_margin": min_margin,
        "known_searches": len(known),
        "impostor_searches": len(impostor),
        "threshold": chosen["threshold"],
        "selected": chosen,
        "curve": curve,
    }
