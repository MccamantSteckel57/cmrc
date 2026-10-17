"""Causal calibration strata from observable market state (Remark 1).

State at time t uses only information accumulated up to t: an expanding-window
drawdown and a moving realized-variance quantile. The evaluation taxonomy in
``regimes`` is a separate object and is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array

STRATUM_NAMES: tuple[str, str, str] = ("stress", "neutral", "calm")


@dataclass(frozen=True)
class StratumConfig:
    drawdown_stress: float = -0.10
    drawdown_calm: float = -0.03
    variance_quantile_stress: float = 0.90
    variance_quantile_calm: float = 0.50
    variance_window: int = 22
    burn_in: int = 66


def expanding_drawdown(prices: FloatArray) -> FloatArray:
    running_peak = np.maximum.accumulate(prices)
    return np.asarray(prices / running_peak - 1.0, dtype=np.float64)


def moving_variance_quantile(realized_variance: FloatArray, window: int, burn_in: int) -> FloatArray:
    values = as_float_array(realized_variance)
    quantiles = np.full(values.size, np.nan, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        history = values[start : index + 1]
        expanding = values[: index + 1]
        if expanding.size < burn_in:
            continue
        quantiles[index] = float(np.mean(expanding <= history[-1]))
    return quantiles


DEFAULT_STRATUM_CONFIG = StratumConfig()


def assign_strata(
    prices: FloatArray,
    realized_variance: FloatArray,
    config: StratumConfig = DEFAULT_STRATUM_CONFIG,
) -> npt.NDArray[np.int64]:
    if prices.size != realized_variance.size:
        raise ValueError("prices and realized variance must have equal length")
    drawdown = expanding_drawdown(as_float_array(prices))
    quantile = moving_variance_quantile(realized_variance, config.variance_window, config.burn_in)
    labels = np.full(prices.size, 1, dtype=np.int64)
    for index in range(prices.size):
        if np.isnan(quantile[index]):
            labels[index] = 1
            continue
        stress = drawdown[index] <= config.drawdown_stress or quantile[index] >= (
            config.variance_quantile_stress
        )
        calm = drawdown[index] >= config.drawdown_calm and quantile[index] <= (
            config.variance_quantile_calm
        )
        labels[index] = 0 if stress else (2 if calm else 1)
    return labels


def stratum_weights(labels: npt.NDArray[np.int64]) -> FloatArray:
    counts = np.bincount(labels, minlength=len(STRATUM_NAMES)).astype(np.float64)
    total = counts.sum()
    if total == 0:
        raise ValueError("cannot compute stratum weights of an empty block")
    return np.asarray(counts / total, dtype=np.float64)


def stratum_slices(labels: npt.NDArray[np.int64]) -> dict[str, npt.NDArray[np.int64]]:
    slices: dict[str, npt.NDArray[np.int64]] = {}
    for index, name in enumerate(STRATUM_NAMES):
        slices[name] = np.flatnonzero(labels == index)
    return slices
