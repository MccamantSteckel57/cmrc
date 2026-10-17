"""Feature builders for the HAR family, return models and credit scorers.

Ref: Sec. 4.1 (train-only features), Table 1 (HAR lags, h = 1).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array

IntArray = npt.NDArray[np.int64]


def trailing_mean(values: FloatArray, window: int) -> FloatArray:
    values = as_float_array(values)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    means = np.full(values.size, np.nan, dtype=np.float64)
    for index in range(window - 1, values.size):
        means[index] = (cumulative[index + 1] - cumulative[index + 1 - window]) / window
    return means


def har_design(realized_variance: FloatArray, horizon: int = 1) -> tuple[FloatArray, FloatArray]:
    values = as_float_array(realized_variance)
    daily = values
    weekly = trailing_mean(values, 5)
    monthly = trailing_mean(values, 22)
    design = np.column_stack([np.ones_like(daily), daily, weekly, monthly])
    target = np.full(values.size, np.nan, dtype=np.float64)
    target[:-horizon] = values[horizon:]
    return design, target


def har_log_design(
    realized_variance: FloatArray, horizon: int = 1
) -> tuple[FloatArray, FloatArray]:
    values = np.log(np.maximum(as_float_array(realized_variance), 1e-16))
    design, target = har_design(values, horizon)
    design = design.copy()
    design[:, 1:] = np.nan_to_num(design[:, 1:], nan=0.0)
    return design, target


def finite_rows(design: FloatArray, target: FloatArray) -> IntArray:
    mask = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    return np.flatnonzero(mask).astype(np.int64)


def absolute_return_design(
    returns: FloatArray, lags: int = 5, vol_window: int = 22
) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = as_float_array(returns)
    level = trailing_mean(np.abs(values), vol_window)
    scale = np.where(np.isnan(level), np.nanmedian(np.abs(values)), level)
    scale = np.maximum(scale, 1e-8)
    columns = [np.ones_like(values)]
    for lag in range(lags):
        shifted = np.full(values.size, np.nan, dtype=np.float64)
        if lag == 0:
            shifted = values.copy()
        else:
            shifted[lag:] = values[:-lag]
        columns.append(shifted)
    columns.append(np.log(scale))
    design = np.column_stack(columns)
    target = np.full(values.size, np.nan, dtype=np.float64)
    target[:-1] = values[1:]
    return design, target, scale


def windowed_batches(
    design: FloatArray, target: FloatArray, window: int, step: int = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    matrix = np.asarray(design, dtype=np.float64)
    response = np.asarray(target, dtype=np.float64)
    features: list[FloatArray] = []
    labels: list[float] = []
    cursor = window
    while cursor < matrix.shape[0]:
        block = matrix[cursor - window : cursor]
        if np.all(np.isfinite(block)) and np.isfinite(response[cursor - 1]):
            features.append(block)
            labels.append(float(response[cursor - 1]))
        cursor += step
    if not features:
        raise ValueError("no complete windows available for the requested length")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
