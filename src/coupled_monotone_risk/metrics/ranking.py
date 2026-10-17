"""Pairwise and set-valued predictive-accuracy comparisons (Sec. 2.3, Sec. 4.2).

Ref: Diebold-Mariano for pairwise loss differences, Hansen-Lunde-Nason model
confidence set for the loss-based ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array


@dataclass(frozen=True)
class PairwiseOutcome:
    statistic: float
    p_value: float
    mean_difference: float
    lag: int


def newey_west_variance(values: FloatArray, lag: int) -> float:
    centred = values - values.mean()
    count = centred.size
    total = float(np.dot(centred, centred) / count)
    for offset in range(1, min(lag, count - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        covariance = float(np.dot(centred[offset:], centred[:-offset]) / count)
        total += 2.0 * weight * covariance
    return total


def diebold_mariano(loss_a: npt.ArrayLike, loss_b: npt.ArrayLike, horizon: int = 1) -> PairwiseOutcome:
    left = as_float_array(loss_a)
    right = as_float_array(loss_b)
    if left.size != right.size:
        raise ValueError("the two loss series must be aligned")
    if horizon < 1:
        raise ValueError("the forecast horizon must be positive")
    difference = left - right
    variance = newey_west_variance(difference, horizon - 1)
    if variance <= 0.0:
        return PairwiseOutcome(0.0, 1.0, float(difference.mean()), horizon - 1)
    statistic = float(difference.mean() / np.sqrt(variance / difference.size))
    p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    return PairwiseOutcome(statistic, p_value, float(difference.mean()), horizon - 1)


@dataclass(frozen=True)
class ConfidenceSet:
    survivors: tuple[int, ...]
    eliminated: tuple[int, ...]
    p_values: FloatArray


def model_confidence_set(
    losses: npt.ArrayLike,
    alpha: float = 0.10,
    bootstrap: int = 500,
    block: int = 10,
    seed: int = 7,
) -> ConfidenceSet:
    """Range-statistic elimination with a circular block bootstrap."""
    matrix = np.asarray(losses, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("losses must be a (n, m) matrix with m >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie inside (0, 1)")
    generator = np.random.default_rng(seed)
    count = matrix.shape[0]
    active = list(range(matrix.shape[1]))
    eliminated: list[int] = []
    p_values = np.zeros(matrix.shape[1], dtype=np.float64)
    while len(active) > 1:
        subset = matrix[:, active]
        means = subset.mean(axis=0)
        variance = np.mean((subset - means) ** 2, axis=0)
        variance = np.where(variance <= 0.0, 1e-12, variance)
        observed = float(np.max(np.abs(means) / np.sqrt(variance / count)))
        draws = np.empty(bootstrap, dtype=np.float64)
        for draw in range(bootstrap):
            starts = generator.integers(0, count, size=int(np.ceil(count / block)))
            indices = np.concatenate(
                [(start + np.arange(block)) % count for start in starts]
            )[:count]
            sample = subset[indices]
            deviations = sample.mean(axis=0) - means
            sample_variance = np.mean((sample - sample.mean(axis=0)) ** 2, axis=0)
            sample_variance = np.where(sample_variance <= 0.0, 1e-12, sample_variance)
            draws[draw] = float(
                np.max(np.abs(deviations) / np.sqrt(sample_variance / count))
            )
        p_value = float(np.mean(draws >= observed))
        p_values[active] = p_value
        if p_value > alpha:
            break
        worst = int(np.argmax(means / np.sqrt(variance / count)))
        eliminated.append(active.pop(worst))
    return ConfidenceSet(survivors=tuple(active), eliminated=tuple(eliminated), p_values=p_values)
