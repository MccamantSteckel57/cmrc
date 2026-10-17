"""Point-forecast accuracy metrics (Table 1, Panel A)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import as_float_array

_FLOOR = 1e-16


def qlike(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    """QLIKE on variances; lower is better and asymmetric by construction."""
    observed = np.maximum(as_float_array(actual), _FLOOR)
    predicted = np.maximum(as_float_array(forecast), _FLOOR)
    ratio = observed / predicted
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mean_squared_error(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    observed = as_float_array(actual)
    predicted = as_float_array(forecast)
    return float(np.mean((observed - predicted) ** 2))


def mean_absolute_deviation(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    observed = np.log(np.maximum(as_float_array(actual), _FLOOR))
    predicted = np.log(np.maximum(as_float_array(forecast), _FLOOR))
    return float(np.mean(np.abs(observed - predicted)))


def directional_accuracy(
    actual: npt.ArrayLike, forecast: npt.ArrayLike, previous: npt.ArrayLike
) -> float:
    observed = as_float_array(actual)
    predicted = as_float_array(forecast)
    reference = as_float_array(previous)
    if not (observed.size == predicted.size == reference.size):
        raise ValueError("actual, forecast and reference must be aligned")
    mask = observed != reference
    if not np.any(mask):
        return float("nan")
    hit = np.sign(predicted[mask] - reference[mask]) == np.sign(observed[mask] - reference[mask])
    return float(np.mean(hit))
