"""Isotonic fits and their sup-norm residuals (Algorithm 1, line 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array, pava_decreasing


@dataclass(frozen=True)
class IsotonicFit:
    x: FloatArray
    y: FloatArray
    fitted: FloatArray

    def __post_init__(self) -> None:
        if not (self.x.size == self.y.size == self.fitted.size):
            raise ValueError("isotonic fit requires matched arrays")
        if np.any(np.diff(self.x) <= 0.0):
            raise ValueError("isotonic abscissa must be strictly increasing")

    @property
    def residual(self) -> float:
        return float(np.max(np.abs(self.y - self.fitted)))

    def evaluate(self, query: npt.ArrayLike) -> FloatArray:
        return np.asarray(np.interp(np.asarray(query, dtype=np.float64), self.x, self.fitted))


def fit_decreasing(x: npt.ArrayLike, y: npt.ArrayLike) -> IsotonicFit:
    abscissa = as_float_array(x)
    ordinate = as_float_array(y)
    return IsotonicFit(x=abscissa, y=ordinate, fitted=pava_decreasing(ordinate))


def fit_increasing(x: npt.ArrayLike, y: npt.ArrayLike) -> IsotonicFit:
    abscissa = as_float_array(x)
    ordinate = as_float_array(y)
    return IsotonicFit(x=abscissa, y=ordinate, fitted=-pava_decreasing(-ordinate))
