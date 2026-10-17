"""HAR family on realized variance (Corsi 2009) with a log variant.

Ref: Sec. 4.2 (HAR benchmarks), Table 1 Panel A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.utils.numeric import FloatArray

_FLOOR = 1e-16


@dataclass
class HarForecaster:
    name: str = "har"
    log_target: bool = False

    def __post_init__(self) -> None:
        self._coefficients: FloatArray | None = None

    def fit(self, design: FloatArray, target: FloatArray) -> None:
        matrix = np.asarray(design, dtype=np.float64)
        response = np.asarray(target, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != response.size:
            raise ValueError("design rows must match the target")
        if matrix.shape[0] < matrix.shape[1] + 2:
            raise ValueError("too few rows to identify the HAR coefficients")
        values = np.log(np.maximum(response, _FLOOR)) if self.log_target else response
        coefficients, *_ = np.linalg.lstsq(matrix, values, rcond=None)
        self._coefficients = np.asarray(coefficients, dtype=np.float64)

    def predict(self, design: FloatArray) -> FloatArray:
        if self._coefficients is None:
            raise RuntimeError("fit must be called before predict")
        matrix = np.asarray(design, dtype=np.float64)
        if matrix.shape[1] != self._coefficients.size:
            raise ValueError("design width does not match the fitted coefficients")
        linear = matrix @ self._coefficients
        if self.log_target:
            return np.asarray(np.exp(np.clip(linear, -30.0, 30.0)), dtype=np.float64)
        return np.asarray(np.maximum(linear, _FLOOR), dtype=np.float64)

    @property
    def coefficients(self) -> FloatArray:
        if self._coefficients is None:
            raise RuntimeError("coefficients are not available before fit")
        return self._coefficients

    def fit_predict(self, train_design: FloatArray, train_target: FloatArray, design: FloatArray) -> FloatArray:
        self.fit(train_design, train_target)
        return self.predict(design)
