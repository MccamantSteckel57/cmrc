"""ARFIMA(1, d, 0) on log realized variance with a GPH estimate of d.

Ref: Sec. 4.2 (ARFIMA benchmark), Table 1 Panel A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from coupled_monotone_risk.utils.numeric import FloatArray

_FLOOR = 1e-16


def fractional_weights(order: float, length: int) -> FloatArray:
    lags = np.arange(length, dtype=np.float64)
    log_weights = gammaln(lags - order) - gammaln(lags + 1.0) - gammaln(-order)
    return np.asarray(np.exp(log_weights), dtype=np.float64)


def fractional_difference(values: FloatArray, weights: FloatArray) -> FloatArray:
    output = np.full(values.size, np.nan, dtype=np.float64)
    for index in range(weights.size - 1, values.size):
        window = values[index - weights.size + 1 : index + 1][::-1]
        output[index] = float(np.dot(weights, window))
    return output


def gph_estimate(values: FloatArray, bandwidth: int | None = None) -> float:
    series = np.asarray(values, dtype=np.float64)
    count = series.size
    bandwidth = int(np.sqrt(count)) if bandwidth is None else bandwidth
    if bandwidth < 4 or bandwidth >= count // 2:
        raise ValueError("the GPH bandwidth must lie between four and n / 2")
    centred = series - series.mean()
    frequencies = 2.0 * np.pi * np.arange(1, bandwidth + 1) / count
    spectrum = np.abs(np.fft.fft(centred)) ** 2 / (2.0 * np.pi * count)
    ordinates = np.log(np.maximum(spectrum[1 : bandwidth + 1], _FLOOR))
    regressor = np.log(4.0 * np.sin(frequencies / 2.0) ** 2)
    design = np.column_stack([np.ones(bandwidth), regressor])
    solution, *_ = np.linalg.lstsq(design, ordinates, rcond=None)
    return float(np.clip(-solution[1], -0.45, 0.45))


@dataclass
class ArfimaForecaster:
    name: str = "arfima"
    truncation: int = 100

    def __post_init__(self) -> None:
        self._order: float = 0.0
        self._intercept: float = 0.0
        self._persistence: float = 0.0
        self._mean: float = 0.0

    def fit(self, series: FloatArray) -> None:
        values = np.log(np.maximum(np.asarray(series, dtype=np.float64), _FLOOR))
        if values.size < 100:
            raise ValueError("ARFIMA needs at least one hundred observations")
        self._order = gph_estimate(values)
        weights = fractional_weights(self._order, self.truncation)
        differenced = fractional_difference(values, weights)
        usable = differenced[~np.isnan(differenced)]
        if usable.size < 20:
            raise ValueError("fractional differencing left too few usable points")
        design = np.column_stack([np.ones(usable.size - 1), usable[:-1]])
        solution, *_ = np.linalg.lstsq(design, usable[1:], rcond=None)
        self._intercept = float(solution[0])
        self._persistence = float(np.clip(solution[1], -0.95, 0.95))
        self._mean = float(usable.mean())

    def filter(self, series: FloatArray) -> FloatArray:
        values = np.log(np.maximum(np.asarray(series, dtype=np.float64), _FLOOR))
        weights = fractional_weights(self._order, self.truncation)
        differenced = fractional_difference(values, weights)
        forecast = np.full(values.size, np.nan, dtype=np.float64)
        for index in range(self.truncation, values.size):
            previous = differenced[index - 1] if index > 0 else self._mean
            if not np.isfinite(previous):
                continue
            next_differenced = self._intercept + self._persistence * previous
            history = values[max(0, index - self.truncation + 1) : index + 1][::-1]
            tail = float(np.dot(weights[1 : history.size], history[: history.size - 1]))
            forecast[index] = np.exp(np.clip(next_differenced - tail, -30.0, 30.0))
        return forecast

    @property
    def order(self) -> float:
        return self._order
