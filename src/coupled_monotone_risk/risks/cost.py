"""Width-cost curves and their marginal costs (Eq. (4)-(6))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.utils.numeric import FloatArray, invert_decreasing, pava_decreasing


@dataclass(frozen=True)
class CostCurve:
    alpha: FloatArray
    width: FloatArray
    fitted: FloatArray
    residual: float

    def __post_init__(self) -> None:
        if not (self.alpha.size == self.width.size == self.fitted.size):
            raise ValueError("cost curve arrays must have equal length")
        if np.any(np.diff(self.alpha) <= 0.0):
            raise ValueError("budget axis must be strictly increasing")
        if np.any(self.fitted < 0.0):
            raise ValueError("widths must be non-negative")

    @classmethod
    def from_curve(
        cls, curve: RiskCurve, budget_grid: FloatArray | None = None
    ) -> CostCurve:
        budgets = (
            np.linspace(0.01, 0.50, 50) if budget_grid is None else np.asarray(budget_grid, float)
        )
        if np.any(np.diff(budgets) <= 0.0):
            raise ValueError("budget grid must be strictly increasing")
        widths = np.asarray(
            [curve.width_at(curve.generalized_inverse(float(budget))) for budget in budgets],
            dtype=np.float64,
        )
        fitted = pava_decreasing(widths)
        residual = float(np.max(np.abs(widths - fitted)))
        return cls(alpha=budgets, width=widths, fitted=fitted, residual=residual)

    def width_at(self, budget: float) -> float:
        return float(np.interp(float(budget), self.alpha, self.fitted))

    def marginal_points(self) -> tuple[FloatArray, FloatArray]:
        midpoints = 0.5 * (self.alpha[1:] + self.alpha[:-1])
        slopes = np.diff(self.fitted) / np.diff(self.alpha)
        marginal = np.maximum(-slopes, 0.0)
        return midpoints, pava_decreasing(marginal)

    def marginal(self, budget: float) -> float:
        midpoints, marginal = self.marginal_points()
        return float(np.interp(float(budget), midpoints, marginal, left=marginal[0], right=marginal[-1]))

    def budget_for_marginal(self, multiplier: float) -> float:
        midpoints, marginal = self.marginal_points()
        level = invert_decreasing(midpoints, marginal, float(multiplier))
        return float(np.clip(level, self.alpha[0], self.alpha[-1]))

    def convexity_violation(self) -> float:
        self.marginal_points()
        raw = np.maximum(-np.diff(self.fitted) / np.diff(self.alpha), 0.0)
        return float(np.mean(np.diff(raw) > 1e-12))


def uniform_width(curve: CostCurve, budget: float) -> float:
    return curve.width_at(budget)
