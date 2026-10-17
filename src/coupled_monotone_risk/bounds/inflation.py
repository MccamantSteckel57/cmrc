"""Budget-inflation relation and the phase-alignment screen (Eq. (10), Fig. 5).

Ref: Sec. 3.4 (how the risk budget grows with K), Sec. 4.7, Figure 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array, spearman

VOID_THRESHOLD = 0.30


@dataclass(frozen=True)
class InflationBounds:
    tasks: FloatArray
    shared: FloatArray
    simultaneous: FloatArray

    @property
    def exponent_shared(self) -> float:
        return 0.0

    @property
    def exponent_simultaneous(self) -> float:
        return 1.0


@dataclass(frozen=True)
class AlignmentScreen:
    correlation: float
    threshold: float

    @property
    def void(self) -> bool:
        return bool(self.correlation < self.threshold)

    @property
    def verdict(self) -> str:
        return "WITHDRAWN" if self.void else "ADMISSIBLE"


def inflation_bounds(task_counts: npt.ArrayLike, budget: float = 0.10) -> InflationBounds:
    counts = np.asarray(task_counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("at least two task counts are required")
    if np.any(counts < 1.0):
        raise ValueError("task counts must be positive")
    shared = np.full(counts.size, budget, dtype=np.float64)
    simultaneous = np.asarray(budget * counts, dtype=np.float64)
    return InflationBounds(tasks=counts, shared=shared, simultaneous=simultaneous)


def measured_exponent(task_counts: npt.ArrayLike, budgets: npt.ArrayLike) -> float:
    """Least-squares slope of log budget on log K; reported, never assumed."""
    counts = as_float_array(task_counts)
    values = as_float_array(budgets)
    if counts.size != values.size or counts.size < 2:
        raise ValueError("matched task counts and budgets are required")
    design = np.column_stack([np.ones_like(counts), np.log(counts)])
    solution, *_ = np.linalg.lstsq(design, np.log(values), rcond=None)
    return float(solution[1])


def residual_rank_correlation(residuals: npt.ArrayLike, timing: npt.ArrayLike) -> float:
    """Rank correlation between two task residuals along the timing axis."""
    matrix = np.asarray(residuals, dtype=np.float64)
    clock = as_float_array(timing)
    if matrix.ndim != 2 or matrix.shape[0] != clock.size:
        raise ValueError("residuals must be (n, tasks) and aligned with the clock")
    if matrix.shape[1] < 2:
        raise ValueError("at least two residual streams are required")
    correlations = [
        spearman(matrix[:, index] - matrix[:, index + 1], clock)
        for index in range(matrix.shape[1] - 1)
    ]
    return float(np.mean(correlations))


def screen_alignment(correlation: float, threshold: float = VOID_THRESHOLD) -> AlignmentScreen:
    return AlignmentScreen(correlation=float(correlation), threshold=float(threshold))
