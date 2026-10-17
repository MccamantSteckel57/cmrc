"""Split, adaptive and quantile baselines on the same protocol (Sec. 4.2).

Ref: Sec. 2.1-2.2, Sec. 4.2, Table 1 Panels B and C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import QuantileRegressor

from coupled_monotone_risk.utils.numeric import (
    FloatArray,
    as_float_array,
    conformal_quantile,
    empirical_quantile,
)


@dataclass(frozen=True)
class OnlineResult:
    levels: FloatArray
    thresholds: FloatArray
    errors: FloatArray
    coverage: FloatArray

    @property
    def mean_level(self) -> float:
        return float(np.mean(self.levels))

    @property
    def final_level(self) -> float:
        return float(self.levels[-1])

    @property
    def mean_coverage(self) -> float:
        return float(np.mean(self.coverage))


class OnlineConformal:
    """One controller with proportional and integral feedback.

    ``eta`` drives the adaptive-conformal update, ``kappa`` adds the integral
    term; setting both terms or the clipping bounds yields the ACI, clipped ACI,
    P-control and PI-control rows of Panel B.
    """

    def __init__(
        self,
        name: str,
        eta: float,
        kappa: float = 0.0,
        clip: tuple[float, float] = (0.0, 1.0),
        window: int = 250,
    ) -> None:
        if eta <= 0.0:
            raise ValueError("the learning rate must be strictly positive")
        if kappa < 0.0:
            raise ValueError("the integral gain cannot be negative")
        if not 0.0 <= clip[0] < clip[1] <= 1.0:
            raise ValueError("clipping bounds must be ordered inside [0, 1]")
        if window < 10:
            raise ValueError("the online window needs at least ten observations")
        self.name = name
        self.eta = float(eta)
        self.kappa = float(kappa)
        self.clip = clip
        self.window = int(window)

    def run(self, residuals: FloatArray, budget: float) -> OnlineResult:
        scores = as_float_array(residuals)
        if scores.size < self.window + 1:
            raise ValueError("the residual stream is shorter than the rolling window")
        level = budget
        levels: list[float] = []
        thresholds: list[float] = []
        errors: list[float] = []
        coverage: list[float] = []
        for index in range(scores.size):
            start = max(0, index - self.window)
            history = scores[start:index]
            if history.size < 10:
                levels.append(level)
                thresholds.append(float(history.max() if history.size else scores[index]))
                errors.append(0.0)
                coverage.append(1.0)
                continue
            threshold = empirical_quantile(history, 1.0 - level)
            observed = float(scores[index] > threshold)
            levels.append(level)
            thresholds.append(threshold)
            errors.append(observed)
            coverage.append(1.0 - observed)
            gap = budget - observed
            updated = level + self.eta * gap + self.kappa * (level - budget)
            level = float(np.clip(updated, self.clip[0], self.clip[1]))
        return OnlineResult(
            levels=np.asarray(levels, dtype=np.float64),
            thresholds=np.asarray(thresholds, dtype=np.float64),
            errors=np.asarray(errors, dtype=np.float64),
            coverage=np.asarray(coverage, dtype=np.float64),
        )


def split_conformal_threshold(residuals: FloatArray, budget: float) -> float:
    scores = as_float_array(residuals)
    if not 0.0 < budget < 1.0:
        raise ValueError("the budget must lie inside (0, 1)")
    return conformal_quantile(scores, 1.0 - budget)


@dataclass(frozen=True)
class ConformalisedQuantile:
    lower: QuantileRegressor
    upper: QuantileRegressor
    offset: float

    @classmethod
    def fit(
        cls,
        features: npt.ArrayLike,
        target: npt.ArrayLike,
        calibration: npt.ArrayLike,
        calibration_target: npt.ArrayLike,
        budget: float,
    ) -> ConformalisedQuantile:
        matrix = np.asarray(features, dtype=np.float64)
        response = np.asarray(target, dtype=np.float64)
        lower = QuantileRegressor(quantile=budget / 2.0, alpha=0.0, solver="highs")
        upper = QuantileRegressor(quantile=1.0 - budget / 2.0, alpha=0.0, solver="highs")
        lower.fit(matrix, response)
        upper.fit(matrix, response)
        calibration_matrix = np.asarray(calibration, dtype=np.float64)
        observed = np.asarray(calibration_target, dtype=np.float64)
        scores = np.maximum(
            lower.predict(calibration_matrix) - observed,
            observed - upper.predict(calibration_matrix),
        )
        return cls(lower=lower, upper=upper, offset=conformal_quantile(scores, 1.0 - budget))

    def interval(self, features: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        matrix = np.asarray(features, dtype=np.float64)
        width_lower = self.lower.predict(matrix) - self.offset
        width_upper = self.upper.predict(matrix) + self.offset
        return (
            np.asarray(width_lower, dtype=np.float64),
            np.asarray(width_upper, dtype=np.float64),
        )
