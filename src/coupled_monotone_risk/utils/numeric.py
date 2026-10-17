"""Numeric primitives shared by the risk curves, allocation and bound layers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def as_float_array(values: Sequence[float] | npt.ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a one-dimensional array, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains non-finite values")
    return array


def check_probability(value: float, name: str) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly inside (0, 1), got {value}")
    return float(value)


def check_weights(weights: Sequence[float]) -> FloatArray:
    array = as_float_array(weights)
    if array.size == 0:
        raise ValueError("weights must be non-empty")
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative")
    total = float(array.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to one, got {total}")
    return array


def empirical_quantile(values: FloatArray, level: float) -> float:
    """Left-continuous inverse CDF at ``level`` with the upper end pinned."""
    if values.size == 0:
        raise ValueError("cannot take a quantile of an empty sample")
    if level <= 0.0:
        return float(values.min())
    if level >= 1.0:
        return float(values.max())
    ordered = np.sort(values)
    index = math.floor(level * (ordered.size - 1))
    return float(ordered[index])


def conformal_quantile(values: FloatArray, level: float) -> float:
    """Split-conformal quantile with the (n + 1) / n finite-sample correction."""
    n = values.size
    if n == 0:
        raise ValueError("cannot take a quantile of an empty sample")
    rank = math.ceil((n + 1) * min(max(level, 0.0), 1.0))
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])


def pava_decreasing(values: FloatArray, weights: FloatArray | None = None) -> FloatArray:
    """Pool-adjacent-violators fit of the closest non-increasing sequence."""
    y = as_float_array(values)
    w = np.ones_like(y) if weights is None else as_float_array(weights)
    if y.size != w.size:
        raise ValueError("values and weights must have equal length")
    if np.any(w <= 0.0):
        raise ValueError("isotonic weights must be strictly positive")
    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []
    for value, weight in zip(y, w, strict=True):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_sizes.append(1)
        while len(block_values) > 1 and block_values[-2] < block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / merged_weight
            merged_size = block_sizes[-2] + block_sizes[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_sizes[-2:] = [merged_size]
    return np.repeat(np.asarray(block_values, dtype=np.float64), block_sizes)


def pava_increasing(values: FloatArray) -> FloatArray:
    return -pava_decreasing(-as_float_array(values))


def bisect_monotone(
    function: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    tolerance: float,
    max_iter: int,
) -> tuple[float, int, bool]:
    """Bisection for a non-increasing ``function`` on ``[lower, upper]``."""
    if lower >= upper:
        raise ValueError("bisection bracket must satisfy lower < upper")
    left, right = float(lower), float(upper)
    for iteration in range(1, max_iter + 1):
        middle = 0.5 * (left + right)
        value = function(middle)
        if abs(value - target) <= tolerance:
            return middle, iteration, True
        if value > target:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right), max_iter, False


def piecewise_linear_slope(x: FloatArray, y: FloatArray) -> FloatArray:
    """Slopes of the piecewise-linear interpolant, one per interval."""
    if x.size != y.size or x.size < 2:
        raise ValueError("need at least two matched points")
    deltas = np.diff(y) / np.diff(x)
    return np.asarray(deltas, dtype=np.float64)


def weighted_mean(values: FloatArray, weights: FloatArray) -> float:
    if values.size != weights.size:
        raise ValueError("values and weights must have equal length")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    return float(np.dot(values, weights) / total)


def stable_softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - float(np.max(logits))
    exponentials = np.exp(shifted)
    return np.asarray(exponentials / exponentials.sum(), dtype=np.float64)


def rank_average(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    return ranks


def spearman(values: FloatArray, other: FloatArray) -> float:
    if values.size != other.size:
        raise ValueError("both series must have equal length")
    left = rank_average(values)
    right = rank_average(other)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def invert_decreasing(x: FloatArray, y: FloatArray, query: float) -> float:
    """First crossing of a non-increasing ``y(x)`` with ``query``, interpolated."""
    if x.size != y.size or x.size < 2:
        raise ValueError("inversion needs at least two matched points")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("abscissa must be strictly increasing")
    if np.any(np.diff(y) > 1e-12):
        raise ValueError("ordinate must be non-increasing")
    if query <= float(y[-1]):
        return float(x[-1])
    if query >= float(y[0]):
        return float(x[0])
    return float(np.interp(-query, -y, x))
