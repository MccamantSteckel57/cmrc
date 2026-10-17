"""Realized GARCH(1,1) with a log-linear variance and measurement equation.

Ref: Sec. 4.2 (realized GARCH benchmark), Table 1 Panel A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from coupled_monotone_risk.utils.numeric import FloatArray

_FLOOR = 1e-16


@dataclass
class RealizedGarchForecaster:
    name: str = "realized_garch"
    max_iter: int = 500

    def __post_init__(self) -> None:
        self._parameters: FloatArray | None = None

    def _recursion(
        self, returns: FloatArray, realized: FloatArray, parameters: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        omega, beta, gamma, xi, delta, _sigma = parameters
        variance = np.empty(returns.size, dtype=np.float64)
        measured = np.empty(returns.size, dtype=np.float64)
        variance[0] = float(np.var(returns) + _FLOOR)
        measured[0] = float(np.mean(np.log(realized)))
        for index in range(1, returns.size):
            variance[index] = np.exp(
                np.clip(omega + beta * np.log(variance[index - 1]) + gamma * measured[index - 1], -30.0, 30.0)
            )
            measured[index] = xi + delta * np.log(variance[index])
        return variance, measured

    def _negative_log_likelihood(
        self, raw: FloatArray, returns: FloatArray, realized: FloatArray
    ) -> float:
        omega, _beta, _gamma, _xi, _delta, sigma = raw
        if sigma <= 1e-6 or np.exp(omega) <= 0.0:
            return 1e10
        variance, measured = self._recursion(returns, realized, np.asarray(raw, dtype=np.float64))
        log_realized = np.log(np.maximum(realized, _FLOOR))
        innovation = log_realized - measured
        return_term = float(np.sum(0.5 * (np.log(variance) + returns**2 / variance)))
        measurement = float(np.sum(0.5 * (np.log(sigma**2) + innovation**2 / sigma**2)))
        return return_term + measurement

    def fit(self, returns: FloatArray, realized: FloatArray) -> None:
        response = np.asarray(returns, dtype=np.float64)
        measure = np.maximum(np.asarray(realized, dtype=np.float64), _FLOOR)
        if response.size != measure.size:
            raise ValueError("returns and realized measures must be aligned")
        if response.size < 60:
            raise ValueError("realized GARCH needs at least sixty observations")
        log_measure = np.log(measure)
        initial = np.asarray(
            [
                float(np.mean(log_measure)) * 0.05,
                0.85,
                0.10,
                float(np.mean(log_measure)) * 0.1,
                0.95,
                float(np.std(np.diff(log_measure)) + 1e-3),
            ],
            dtype=np.float64,
        )
        bounds = [
            (-5.0, 2.0),
            (0.0, 0.999),
            (-0.9, 0.9),
            (-2.0, 2.0),
            (0.2, 1.5),
            (1e-5, 2.0),
        ]
        result = optimize.minimize(
            self._negative_log_likelihood,
            initial,
            args=(response, measure),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter},
        )
        self._parameters = np.asarray(result.x, dtype=np.float64)

    def filter(self, returns: FloatArray, realized: FloatArray) -> FloatArray:
        if self._parameters is None:
            raise RuntimeError("fit must be called before filter")
        response = np.asarray(returns, dtype=np.float64)
        measure = np.maximum(np.asarray(realized, dtype=np.float64), _FLOOR)
        if response.size != measure.size:
            raise ValueError("returns and realized measures must be aligned")
        variance, _ = self._recursion(response, measure, self._parameters)
        return np.asarray(variance, dtype=np.float64)

    @property
    def parameters(self) -> FloatArray:
        if self._parameters is None:
            raise RuntimeError("parameters are not available before fit")
        return self._parameters
