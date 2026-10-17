"""Discrimination and calibration metrics for the credit arm (Table 1, Panel D)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array, rank_average


def _binary_inputs(target: npt.ArrayLike, score: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    labels = np.asarray(target, dtype=np.int64)
    values = as_float_array(score)
    if labels.size != values.size:
        raise ValueError("labels and scores must be aligned")
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError("binary metrics need exactly two label values")
    return values, np.asarray(labels == unique[-1], dtype=np.float64)


def auc_roc(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    values, positive = _binary_inputs(target, score)
    ranks = rank_average(values)
    count = float(positive.sum())
    negative = float(positive.size - count)
    if count == 0.0 or negative == 0.0:
        return float("nan")
    return float((ranks[positive == 1.0].sum() - count * (count + 1.0) / 2.0) / (count * negative))


def average_precision(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    values, positive = _binary_inputs(target, score)
    order = np.argsort(-values, kind="mergesort")
    ordered = positive[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, ordered.size + 1)
    total = float(ordered.sum())
    if total == 0.0:
        return float("nan")
    return float(np.sum(precision * ordered) / total)


def ks_statistic(target: npt.ArrayLike, score: npt.ArrayLike) -> float:
    values, positive = _binary_inputs(target, score)
    order = np.argsort(values, kind="mergesort")
    ordered = positive[order]
    positives = float(ordered.sum())
    negatives = float(ordered.size - positives)
    if positives == 0.0 or negatives == 0.0:
        return float("nan")
    cdf_positive = np.cumsum(ordered) / positives
    cdf_negative = np.cumsum(1.0 - ordered) / negatives
    return float(np.max(np.abs(cdf_positive - cdf_negative)))


def brier_score(target: npt.ArrayLike, probability: npt.ArrayLike) -> float:
    labels = np.asarray(target, dtype=np.float64)
    values = as_float_array(probability)
    if labels.size != values.size:
        raise ValueError("labels and probabilities must be aligned")
    return float(np.mean((values - labels) ** 2))


def expected_calibration_error(
    target: npt.ArrayLike, probability: npt.ArrayLike, bins: int = 10
) -> float:
    labels = np.asarray(target, dtype=np.float64)
    values = as_float_array(probability)
    if labels.size != values.size:
        raise ValueError("labels and probabilities must be aligned")
    if bins < 2:
        raise ValueError("at least two bins are required")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (values >= lower) & (values < upper if index < bins - 1 else values <= upper)
        if not np.any(mask):
            continue
        gap = abs(float(values[mask].mean()) - float(labels[mask].mean()))
        total += gap * float(mask.mean())
    return float(total)


def prediction_set_size(mask: npt.NDArray[np.bool_]) -> float:
    if mask.ndim != 2:
        raise ValueError("set masks must be (n, classes)")
    return float(np.mean(np.maximum(mask.sum(axis=1), 1)))
