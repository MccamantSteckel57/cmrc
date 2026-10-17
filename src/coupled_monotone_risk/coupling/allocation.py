"""Equalised marginal width cost allocation (Eq. (4)-(6), Proposition 1).

Ref: Sec. 3.2, Algorithm 1 lines 4-16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.utils.numeric import (
    FloatArray,
    check_weights,
    invert_decreasing,
    pava_decreasing,
)

MULTIPLIER_GRID = 513


@dataclass(frozen=True)
class AllocationResult:
    shares: FloatArray
    budgets: FloatArray
    multiplier: float
    iterations: int
    converged: bool
    allocated_objective: float
    uniform_objective: float
    budget_residual: float

    @property
    def objective_ratio(self) -> float:
        if self.uniform_objective <= 0.0:
            return float("nan")
        return float(self.allocated_objective / self.uniform_objective)

    @property
    def share_sum_residual(self) -> float:
        return float(abs(np.dot(self.shares, np.ones_like(self.shares)) - 1.0))


def _total_budget(multiplier: float, curves: tuple[CostCurve, ...], weights: FloatArray) -> float:
    budgets = np.asarray([curve.budget_for_marginal(multiplier) for curve in curves])
    return float(np.dot(weights, budgets))


def _objective(curves: tuple[CostCurve, ...], weights: FloatArray, budgets: FloatArray) -> float:
    return float(
        np.dot(
            weights,
            np.asarray(
                [curve.width_at(b) for curve, b in zip(curves, budgets, strict=True)]
            ),
        )
    )


def _solve_multiplier(
    curves: tuple[CostCurve, ...], weights: FloatArray, budget: float
) -> tuple[float, int]:
    """Root of Eq. (5) on a multiplier grid; the total budget is a step function."""
    upper = max(curve.marginal(curve.alpha[0]) for curve in curves)
    if upper <= 0.0:
        upper = 1.0
    grid = np.linspace(0.0, upper * 1.05, MULTIPLIER_GRID)
    totals = np.asarray([_total_budget(value, curves, weights) for value in grid])
    totals = pava_decreasing(totals)
    multiplier = invert_decreasing(grid, totals, budget)
    return multiplier, int(np.searchsorted(-grid, -multiplier))


def equalised_marginal_allocation(
    curves: tuple[CostCurve, ...],
    weights: FloatArray,
    budget: float,
    tolerance: float = 1e-3,
    max_iter: int = MULTIPLIER_GRID,
) -> AllocationResult:
    """Solve Eq. (5) for one multiplier and return the induced shares ``c_k``.

    The multiplier is located on a fixed grid because the total-budget map is a
    step function of the multiplier; shares are then rescaled so that Eq. (6)'s
    identity ``sum_k w_k c_k = 1`` holds exactly, and the pre-scaling residual is
    reported rather than hidden.
    """
    checked = check_weights(list(np.asarray(weights, dtype=np.float64)))
    if len(curves) != checked.size:
        raise ValueError("one cost curve per task weight is required")
    if not 0.0 < budget < 1.0:
        raise ValueError("the aggregate budget must lie inside (0, 1)")
    multiplier, steps = _solve_multiplier(curves, checked, budget)
    budgets = np.asarray([curve.budget_for_marginal(multiplier) for curve in curves])
    residual = abs(float(np.dot(checked, budgets)) - budget)
    shares = budgets / budget
    normaliser = float(np.dot(checked, shares))
    if normaliser <= 0.0:
        raise ValueError("the allocation produced a non-positive share total")
    shares = np.asarray(shares / normaliser, dtype=np.float64)
    budgets = np.asarray(shares * budget, dtype=np.float64)
    uniform = _objective(curves, checked, np.full(checked.size, budget))
    allocated = _objective(curves, checked, budgets)
    return AllocationResult(
        shares=shares,
        budgets=budgets,
        multiplier=float(multiplier),
        iterations=int(min(steps, max_iter)),
        converged=bool(residual <= tolerance),
        allocated_objective=allocated,
        uniform_objective=uniform,
        budget_residual=residual,
    )


def uniform_allocation(count: int, budget: float) -> AllocationResult:
    if count < 1:
        raise ValueError("at least one task is required")
    shares = np.full(count, 1.0, dtype=np.float64)
    budgets = np.full(count, budget, dtype=np.float64)
    return AllocationResult(
        shares=shares,
        budgets=budgets,
        multiplier=float("nan"),
        iterations=0,
        converged=True,
        allocated_objective=float("nan"),
        uniform_objective=float("nan"),
        budget_residual=0.0,
    )


def marginal_cost_spread(curves: tuple[CostCurve, ...], budget: float = 0.10) -> float:
    """Ratio of the largest to the smallest marginal cost; one means proportional."""
    marginal = np.asarray([curve.marginal(budget) for curve in curves], dtype=np.float64)
    positive = marginal[marginal > 0.0]
    if positive.size < 2:
        return 1.0
    return float(np.max(positive) / np.min(positive))
