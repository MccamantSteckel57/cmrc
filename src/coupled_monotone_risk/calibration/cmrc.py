"""CMRC calibration: one shared scalar through K coupling maps (Algorithm 1).

Ref: Sec. 3.2-3.3, Algorithm 1, Theorem 1, Remark 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.coupling.allocation import (
    AllocationResult,
    equalised_marginal_allocation,
)
from coupled_monotone_risk.coupling.composite import (
    CompositeRisk,
    CrossingResult,
    StratifiedView,
    find_crossing,
)
from coupled_monotone_risk.coupling.maps import CouplingMap, build_maps
from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.risks.family import NestedFamily
from coupled_monotone_risk.utils.numeric import FloatArray, check_probability

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TaskInput:
    name: str
    family: NestedFamily
    weight: float


@dataclass(frozen=True)
class TaskCertificate:
    name: str
    weight: float
    share: float
    budget: float
    level: float
    task_level: float
    threshold: float
    calibrated_risk: float
    width: float


@dataclass(frozen=True)
class Diagnostics:
    cost_residuals: FloatArray
    allocated_width: float
    uniform_width: float
    multiplier: float
    iterations: int
    converged: bool
    allocation_residual: float
    non_monotone_share: float
    right_continuous: bool
    monotonicity_share: float
    injective_share: FloatArray
    shift_band: float
    effective_size: float
    stratum_weights: FloatArray
    stratum_losses: FloatArray
    feasibility: bool
    grid_size: int


@dataclass(frozen=True)
class CalibrationResult:
    level: float
    budget: float
    tasks: tuple[TaskCertificate, ...]
    diagnostics: Diagnostics
    composite: CompositeRisk
    maps: tuple[CouplingMap, ...]
    achieved_risk: float
    shifted_risk: float

    @property
    def feasible(self) -> bool:
        return self.diagnostics.feasibility

    def thresholds(self) -> dict[str, float]:
        return {task.name: task.threshold for task in self.tasks}

    def widths(self) -> dict[str, float]:
        return {task.name: task.width for task in self.tasks}

    def joint_width(self) -> float:
        return float(sum(task.weight * task.width for task in self.tasks))

    def certificate_budget(self) -> float:
        return self.budget + self.diagnostics.shift_band


class CmrcCalibrator:
    """Algorithm 1 with phases A, B, C, D and E kept as separate steps."""

    def __init__(
        self,
        budget: float = 0.10,
        grid_size: int = 1001,
        tolerance: float = 1e-10,
        max_iter: int = 200,
        shift_band: float = 0.0,
        bound: float = 1.0,
        strata: int = 3,
    ) -> None:
        self.budget = check_probability(budget, "budget")
        if grid_size < 21:
            raise ValueError("the shared level grid needs at least twenty-one points")
        if strata < 1:
            raise ValueError("at least one stratum is required")
        self.grid_size = int(grid_size)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        if not 0.0 <= shift_band < 1.0 - budget:
            raise ValueError("the shift band must keep the certificate level inside (0, 1)")
        self.shift_band = float(shift_band)
        self.bound = float(bound)
        self.strata = int(strata)

    def fit(
        self,
        tasks: Sequence[TaskInput],
        strata_labels: Sequence[IntArray] | None = None,
        test_strata_labels: Sequence[IntArray] | None = None,
        shares: FloatArray | None = None,
    ) -> CalibrationResult:
        if not tasks:
            raise ValueError("CMRC needs at least one task")
        weights = np.asarray([task.weight for task in tasks], dtype=np.float64)
        weights = weights / weights.sum()
        names = tuple(task.name for task in tasks)

        curves = tuple(RiskCurve.from_family(task.family, self.grid_size) for task in tasks)
        costs = tuple(CostCurve.from_curve(curve) for curve in curves)
        allocation: AllocationResult = (
            equalised_marginal_allocation(
                costs, weights, self.budget, tolerance=self.tolerance, max_iter=self.max_iter
            )
            if shares is None
            else _fixed_shares(costs, weights, self.budget, shares)
        )
        maps = build_maps(names, curves, allocation.shares, self.budget)
        composite = CompositeRisk(maps=maps, weights=weights, budget=self.budget, bound=self.bound)

        view = self._view(composite, tasks, strata_labels)
        risk_fn = composite.risk if view is None else view.risk
        sample_size = (
            int(min(task.family.size for task in tasks))
            if view is None
            else int(min(task_labels.size for task_labels in strata_labels or ()))
        )
        crossing: CrossingResult = find_crossing(
            risk_fn,
            budget=self.budget,
            sample_size=sample_size,
            bound=self.bound,
            grid_size=self.grid_size,
        )

        certificates = tuple(
            self._certificate(task, mapping, crossing.level)
            for task, mapping in zip(tasks, maps, strict=True)
        )
        band = self.shift_band
        if view is not None and test_strata_labels is not None:
            target = self._test_stratum_weights(view, test_strata_labels)
            band = self.shift_band + view.weight_error(target)
        diagnostics = Diagnostics(
            cost_residuals=np.asarray([cost.residual for cost in costs], dtype=np.float64),
            allocated_width=allocation.allocated_objective,
            uniform_width=allocation.uniform_objective,
            multiplier=allocation.multiplier,
            iterations=allocation.iterations,
            converged=allocation.converged,
            allocation_residual=abs(float(np.dot(weights, allocation.budgets)) - self.budget),
            non_monotone_share=crossing.non_monotone_share,
            right_continuous=crossing.right_continuous,
            monotonicity_share=float(np.mean([mapping.monotone_profile()[0] for mapping in maps])),
            injective_share=np.asarray(
                [mapping.injective_share() for mapping in maps], dtype=np.float64
            ),
            shift_band=band,
            effective_size=float(sample_size),
            stratum_weights=(
                np.asarray([], dtype=np.float64) if view is None else view.weights.copy()
            ),
            stratum_losses=(
                np.asarray([], dtype=np.float64)
                if view is None
                else np.asarray(view.stratum_risk(crossing.level), dtype=np.float64)
            ),
            feasibility=crossing.feasible,
            grid_size=self.grid_size,
        )
        return CalibrationResult(
            level=crossing.level,
            budget=self.budget,
            tasks=certificates,
            diagnostics=diagnostics,
            composite=composite,
            maps=maps,
            achieved_risk=crossing.achieved_risk,
            shifted_risk=crossing.shifted_risk,
        )

    def _view(
        self,
        composite: CompositeRisk,
        tasks: Sequence[TaskInput],
        labels: Sequence[IntArray] | None,
    ) -> StratifiedView | None:
        if labels is None:
            return None
        if len(labels) != len(tasks):
            raise ValueError("one stratum label vector per task is required")
        return StratifiedView.from_labels(composite, tuple(labels), self.strata)

    def _test_stratum_weights(
        self, view: StratifiedView, test_labels: Sequence[IntArray]
    ) -> FloatArray:
        counts = np.zeros(view.weights.size, dtype=np.float64)
        for weight, labels in zip(view.composite.weights, test_labels, strict=True):
            if labels.size == 0:
                continue
            for index in range(counts.size):
                counts[index] += float(weight) * float(np.mean(labels == index))
        total = counts.sum()
        return np.asarray(counts / total if total > 0 else counts, dtype=np.float64)

    def _certificate(self, task: TaskInput, mapping: CouplingMap, level: float) -> TaskCertificate:
        task_level = mapping(level)
        return TaskCertificate(
            name=task.name,
            weight=float(task.weight),
            share=float(mapping.share),
            budget=float(mapping.task_budget),
            level=float(level),
            task_level=float(task_level),
            threshold=float(task.family.threshold(task_level)),
            calibrated_risk=float(task.family.risk(task_level)),
            width=float(task.family.width(task_level)),
        )


def _fixed_shares(
    curves: tuple[CostCurve, ...], weights: FloatArray, budget: float, shares: FloatArray
) -> AllocationResult:
    if shares.size != len(curves):
        raise ValueError("one fixed share per task is required")
    if np.any(shares <= 0.0):
        raise ValueError("fixed shares must be strictly positive")
    budgets = np.asarray(shares * budget, dtype=np.float64)
    allocated = float(
        np.dot(
            weights,
            np.asarray(
                [curve.width_at(value) for curve, value in zip(curves, budgets, strict=True)]
            ),
        )
    )
    uniform = float(
        np.dot(
            weights,
            np.asarray([curve.width_at(budget) for curve in curves]),
        )
    )
    return AllocationResult(
        shares=np.asarray(shares, dtype=np.float64),
        budgets=budgets,
        multiplier=float("nan"),
        iterations=0,
        converged=True,
        allocated_objective=allocated,
        uniform_objective=uniform,
        budget_residual=abs(float(np.dot(weights, budgets)) - budget),
    )
