"""Conditional heteroskedastic autoregression on log realized variance.

Ref: Sec. 4.2 (CHAR benchmark), Table 1 Panel A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from coupled_monotone_risk.utils.numeric import FloatArray

_FLOOR = 1e-16


@dataclass
class CharForecaster:
    name: str = "char"
    max_iter: int = 400

    def __post_init__(self) -> None:
        self._parameters: FloatArray | None = None

    def _unpack(self, raw: FloatArray) -> tuple[float, float, float, float, float]:
        intercept, persistence, omega, alpha, beta = raw
        return float(intercept), float(persistence), float(omega), float(alpha), float(beta)

    def _filter(
        self, values: FloatArray, parameters: tuple[float, float, float, float, float]
    ) -> tuple[FloatArray, FloatArray]:
        intercept, persistence, omega, alpha, beta = parameters
        level = np.empty(values.size, dtype=np.float64)
        variance = np.empty(values.size, dtype=np.float64)
        level[0] = float(np.mean(values))
        variance[0] = float(np.var(values) + _FLOOR)
        for index in range(1, values.size):
            shock = values[index - 1] - level[index - 1]
            variance[index] = omega + alpha * shock**2 + beta * variance[index - 1]
            level[index] = intercept + persistence * values[index - 1]
        return level, variance

    def _negative_log_likelihood(self, raw: FloatArray, values: FloatArray) -> float:
        intercept, persistence, omega, alpha, beta = self._unpack(raw)
        if alpha + beta >= 1.0 or omega <= 0.0 or alpha < 0.0 or beta < 0.0:
            return 1e10
        level, variance = self._filter(values, (intercept, persistence, omega, alpha, beta))
        shock = values - level
        total = float(np.sum(np.log(variance) + shock**2 / variance))
        return 0.5 * total

    def fit(self, series: FloatArray) -> None:
        values = np.log(np.maximum(np.asarray(series, dtype=np.float64), _FLOOR))
        if values.size < 50:
            raise ValueError("CHAR needs at least fifty observations")
        mean = float(values.mean())
        centred = values - mean
        persistence = float(np.dot(centred[1:], centred[:-1]) / max(np.dot(centred[:-1], centred[:-1]), _FLOOR))
        residuals = centred[1:] - persistence * centred[:-1]
        initial = np.asarray(
            [mean * (1.0 - persistence), persistence, float(np.var(residuals)) * 0.1, 0.08, 0.88],
            dtype=np.float64,
        )
        bounds = [
            (None, None),
            (-0.999, 0.999),
            (1e-10, None),
            (1e-6, 0.6),
            (1e-6, 0.999),
        ]
        result = optimize.minimize(
            self._negative_log_likelihood,
            initial,
            args=(values,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self._parameters = np.asarray(result.x, dtype=np.float64)

    def filter(self, series: FloatArray) -> FloatArray:
        if self._parameters is None:
            raise RuntimeError("fit must be called before filter")
        values = np.log(np.maximum(np.asarray(series, dtype=np.float64), _FLOOR))
        level, variance = self._filter(values, self._unpack(self._parameters))
        forecast = np.exp(np.clip(level + 0.5 * variance, -30.0, 30.0))
        return np.asarray(forecast, dtype=np.float64)

    @property
    def parameters(self) -> FloatArray:
        if self._parameters is None:
            raise RuntimeError("parameters are not available before fit")
        return self._parameters
