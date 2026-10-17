"""Exponential stratum reweighting and its long-run statement (Sec. 3.4).

Ref: Sec. 3.4 (adaptive variant and the long-run bound).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.utils.numeric import FloatArray


@dataclass(frozen=True)
class LongRunBound:
    average_risk: float
    budget: float
    learning_rate: float
    drift: float

    @property
    def excess(self) -> float:
        return self.average_risk - self.budget

    @property
    def allowance(self) -> float:
        if self.learning_rate <= 0.0:
            raise ValueError("learning rate must be positive to form the allowance")
        return float(self.drift / self.learning_rate)


class AdaptiveStratumWeights:
    """``w_r(t+1) ∝ w_r(t) exp(-a * loss_r(t))`` with normalisation."""

    def __init__(self, initial: FloatArray, learning_rate: float) -> None:
        weights = np.asarray(initial, dtype=np.float64)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("initial weights must be a non-empty vector")
        if np.any(weights < 0.0):
            raise ValueError("initial weights must be non-negative")
        if learning_rate <= 0.0:
            raise ValueError("the learning rate must be strictly positive")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("initial weights must have a positive sum")
        self._weights = weights / total
        self._rate = float(learning_rate)
        self._history: list[FloatArray] = [self._weights.copy()]

    @property
    def weights(self) -> FloatArray:
        return self._weights.copy()

    @property
    def history(self) -> tuple[FloatArray, ...]:
        return tuple(self._history)

    def update(self, losses: FloatArray) -> FloatArray:
        values = np.asarray(losses, dtype=np.float64)
        if values.size != self._weights.size:
            raise ValueError("one loss per stratum is required")
        if np.any(values < 0.0):
            raise ValueError("losses must be non-negative")
        updated = self._weights * np.exp(-self._rate * values)
        total = float(updated.sum())
        if total <= 0.0:
            updated = np.full_like(self._weights, 1.0 / self._weights.size)
        else:
            updated = updated / total
        self._weights = np.asarray(updated, dtype=np.float64)
        self._history.append(self._weights.copy())
        return self._weights

    def long_run(self, average_risk: float, budget: float, drift: float) -> LongRunBound:
        return LongRunBound(
            average_risk=float(average_risk),
            budget=float(budget),
            learning_rate=self._rate,
            drift=float(drift),
        )


def time_averaged_risk(risks: FloatArray) -> float:
    values = np.asarray(risks, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot average an empty risk sequence")
    return float(np.mean(values))


def drift_term(weights: tuple[FloatArray, ...]) -> float:
    if len(weights) < 2:
        return 0.0
    increments = [
        float(np.abs(weights[index + 1] - weights[index]).sum())
        for index in range(len(weights) - 1)
    ]
    return float(np.mean(increments))
