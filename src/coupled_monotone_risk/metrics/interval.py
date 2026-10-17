"""Interval metrics and binomial intervals (Table 1, Table 4.6)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import as_float_array


def coverage(lower: npt.ArrayLike, upper: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    low = as_float_array(lower)
    high = as_float_array(upper)
    observed = as_float_array(actual)
    if not (low.size == high.size == observed.size):
        raise ValueError("interval ends and outcomes must be aligned")
    if np.any(low > high):
        raise ValueError("lower interval end exceeds the upper end")
    return float(np.mean((observed >= low) & (observed <= high)))


def mean_width(lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    low = as_float_array(lower)
    high = as_float_array(upper)
    if low.size != high.size:
        raise ValueError("interval ends must be aligned")
    return float(np.mean(high - low))


def winkler_score(
    lower: npt.ArrayLike, upper: npt.ArrayLike, actual: npt.ArrayLike, alpha: float
) -> float:
    low = as_float_array(lower)
    high = as_float_array(upper)
    observed = as_float_array(actual)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie inside (0, 1)")
    width = high - low
    penalty = (2.0 / alpha) * (
        np.maximum(low - observed, 0.0) + np.maximum(observed - high, 0.0)
    )
    return float(np.mean(width + penalty))


def pinball(actual: npt.ArrayLike, quantile: npt.ArrayLike, tau: float) -> float:
    observed = as_float_array(actual)
    predicted = as_float_array(quantile)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie inside (0, 1)")
    residual = observed - predicted
    return float(np.mean(np.maximum(tau * residual, (tau - 1.0) * residual)))


def crps_from_pinball(
    actual: npt.ArrayLike, quantiles: npt.ArrayLike, taus: npt.ArrayLike
) -> float:
    observed = as_float_array(actual)
    levels = as_float_array(taus)
    columns = np.asarray(quantiles, dtype=np.float64)
    if columns.ndim != 2 or columns.shape[0] != observed.size:
        raise ValueError("quantile forecasts must be (n, |taus|)")
    if columns.shape[1] != levels.size:
        raise ValueError("one quantile column per tau is required")
    scores = [pinball(observed, columns[:, index], float(levels[index])) for index in range(levels.size)]
    return float(2.0 * np.mean(scores))


def wald_interval(hits: npt.ArrayLike, z_value: float = 1.959963984540054) -> tuple[float, float]:
    values = np.asarray(hits, dtype=np.float64)
    count = values.size
    if count == 0:
        raise ValueError("cannot form an interval from an empty sample")
    proportion = float(np.mean(values))
    standard_error = math.sqrt(max(proportion * (1.0 - proportion), 1e-12) / count)
    lower = max(0.0, proportion - z_value * standard_error)
    upper = min(1.0, proportion + z_value * standard_error)
    return float(lower), float(upper)
