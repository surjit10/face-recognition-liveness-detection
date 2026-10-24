"""Dependency-light binary PAD metrics and validation-threshold selection."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    predicted_live = scores >= threshold
    actual_live = labels == 1
    return {
        "tp": int(np.sum(predicted_live & actual_live)),
        "fp": int(np.sum(predicted_live & ~actual_live)),
        "tn": int(np.sum(~predicted_live & ~actual_live)),
        "fn": int(np.sum(~predicted_live & actual_live)),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    attack_total = counts["fp"] + counts["tn"]
    live_total = counts["fn"] + counts["tp"]
    if attack_total == 0 or live_total == 0:
        raise RuntimeError("Both PAD classes are required for metrics")
    apcer = counts["fp"] / attack_total
    bpcer = counts["fn"] / live_total
    return {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2.0}


def _validated(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1 or len(labels) == 0:
        raise RuntimeError("Metric inputs must be equal non-empty one-dimensional arrays")
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Scores must be finite and both binary labels must be present")
    return labels, scores


def select_eer_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Choose the validation threshold where APCER and BPCER are closest.

    This is O(n log n), including tied-score handling. A higher threshold wins
    deterministic ties after absolute-rate difference and ACER.
    """

    labels, scores = _validated(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    live_total = int(np.sum(labels == 1))
    attack_total = len(labels) - live_total
    counts = {"tp": 0, "fp": 0, "tn": attack_total, "fn": live_total}
    above_max = float(np.nextafter(ordered_scores[0], math.inf))
    candidates: list[tuple[float, dict[str, int]]] = [(above_max, counts.copy())]
    index = 0
    while index < len(labels):
        end = index + 1
        while end < len(labels) and ordered_scores[end] == ordered_scores[index]:
            end += 1
        group = ordered_labels[index:end]
        live = int(np.sum(group == 1))
        attack = len(group) - live
        counts["tp"] += live
        counts["fn"] -= live
        counts["fp"] += attack
        counts["tn"] -= attack
        candidates.append((float(ordered_scores[index]), counts.copy()))
        index = end

    def key(item: tuple[float, dict[str, int]]) -> tuple[float, float, float]:
        metric = rates(item[1])
        return (abs(metric["apcer"] - metric["bpcer"]), metric["acer"], -item[0])

    threshold, best_counts = min(candidates, key=key)
    return {"threshold": threshold, "counts": best_counts, **rates(best_counts)}


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = _validated(labels, scores)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    combined = np.concatenate((positive, negative))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    index = 0
    while index < len(combined):
        end = index + 1
        while end < len(combined) and combined[order[end]] == combined[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    rank_sum = float(np.sum(ranks[: len(positive)]))
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (
        len(positive) * len(negative)
    )
