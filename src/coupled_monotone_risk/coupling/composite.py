"""Composite risk on the shared threshold axis (Eq. (1), (3), Algorithm 1 C-E)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.coupling.maps import CouplingMap
from coupled_monotone_risk.risks.family import Mask
from coupled_monotone_risk.utils.numeric import FloatArray

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CrossingResult:
    level: float
    achieved_risk: float
    shifted_risk: float
    budget: float
    feasible: bool
    grid_size: int
    non_monotone_share: float
    right_continuous: bool
    effective_size: int


@dataclass(frozen=True)
class CompositeRisk:
    maps: tuple[CouplingMap, ...]
    weights: FloatArray
    budget: float
    bound: float = 1.0

    def __post_init__(self) -> None:
        if len(self.maps) != self.weights.size:
            raise ValueError("one coupling map per task weight is required")
        if abs(float(self.weights.sum()) - 1.0) > 1e-9:
            raise ValueError("task weights must sum to one")
        if not 0.0 < self.budget < 1.0:
            raise ValueError("the aggregate budget must lie inside (0, 1)")

    def risk(self, level: float, mask: Mask | None = None) -> float:
        contributions = np.asarray(
            [mapping.task_risk(level) if mask is None else mapping.task_risk(level, mask) for mapping in self.maps],
            dtype=np.float64,
        )
        return float(np.dot(self.weights, contributions))

    def risk_with_masks(self, level: float, masks: tuple[Mask, ...]) -> float:
        if len(masks) != len(self.maps):
            raise ValueError("one mask per task is required")
        contributions = np.asarray(
            [
                mapping.task_risk(level, mask)
                for mapping, mask in zip(self.maps, masks, strict=True)
            ],
            dtype=np.float64,
        )
        return float(np.dot(self.weights, contributions))

    def task_contributions(self, level: float) -> dict[str, float]:
        return {mapping.task: mapping.task_risk(level) for mapping in self.maps}

    def risk_vector(self, levels: FloatArray) -> FloatArray:
        return np.asarray([self.risk(float(level)) for level in levels], dtype=np.float64)

    def non_monotone_share(self, levels: FloatArray) -> float:
        values = self.risk_vector(levels)
        return float(np.mean(np.diff(values) > 1e-9))


def find_crossing(
    risk: Callable[[float], float],
    budget: float,
    sample_size: int,
    bound: float = 1.0,
    grid_size: int = 1001,
) -> CrossingResult:
    """Lowest grid level meeting the shifted criterion of Algorithm 1 line 21."""
    if sample_size < 2:
        raise ValueError("the calibration block needs at least two observations")
    if not 0.0 < budget < 1.0:
        raise ValueError("the budget must lie inside (0, 1)")
    levels = np.linspace(0.0, 1.0, grid_size)
    values = np.asarray([risk(float(level)) for level in levels], dtype=np.float64)
    shifted = (sample_size / (sample_size + 1.0)) * values + bound / (sample_size + 1.0)
    feasible = np.flatnonzero(shifted <= budget + 1e-12)
    non_monotone = float(np.mean(np.diff(values) > 1e-9))
    if feasible.size == 0:
        return CrossingResult(
            level=float(levels[-1]),
            achieved_risk=float(values[-1]),
            shifted_risk=float(shifted[-1]),
            budget=budget,
            feasible=False,
            grid_size=grid_size,
            non_monotone_share=non_monotone,
            right_continuous=False,
            effective_size=sample_size,
        )
    index = int(feasible[0])
    nearby = float(risk(float(min(1.0, levels[index] + 1.0 / grid_size))))
    return CrossingResult(
        level=float(levels[index]),
        achieved_risk=float(values[index]),
        shifted_risk=float(shifted[index]),
        budget=budget,
        feasible=True,
        grid_size=grid_size,
        non_monotone_share=non_monotone,
        right_continuous=bool(abs(nearby - values[index]) <= 1e-6),
        effective_size=sample_size,
    )


@dataclass(frozen=True)
class StratifiedView:
    """Regime-stratified composite risk with block weights (Remark 1)."""

    composite: CompositeRisk
    masks: tuple[tuple[Mask, ...], ...]
    weights: FloatArray

    def __post_init__(self) -> None:
        if len(self.masks) != len(self.composite.maps):
            raise ValueError("one mask tuple per task is required")
        if abs(float(self.weights.sum()) - 1.0) > 1e-9:
            raise ValueError("stratum weights must sum to one")
        for task_masks in self.masks:
            if len(task_masks) != self.weights.size:
                raise ValueError("every task needs one mask per stratum")

    @classmethod
    def from_labels(
        cls, composite: CompositeRisk, labels: tuple[IntArray, ...], strata: int
    ) -> StratifiedView:
        if len(labels) != len(composite.maps):
            raise ValueError("one label vector per task is required")
        masks: list[tuple[Mask, ...]] = []
        counts = np.zeros(strata, dtype=np.float64)
        for weight, mapping, task_labels in zip(
            composite.weights, composite.maps, labels, strict=True
        ):
            if task_labels.size != mapping.curve.family.size:
                raise ValueError(f"task {mapping.task} has mismatched stratum labels")
            if task_labels.size == 0:
                raise ValueError(f"task {mapping.task} has an empty calibration block")
            per_task = tuple(
                np.asarray(task_labels == index, dtype=np.bool_) for index in range(strata)
            )
            if sum(int(mask.sum()) for mask in per_task) != task_labels.size:
                raise ValueError("stratum labels must lie inside the declared range")
            masks.append(per_task)
            for index, mask in enumerate(per_task):
                counts[index] += float(weight) * float(mask.sum()) / task_labels.size
        total = counts.sum()
        if total <= 0.0:
            raise ValueError("stratum weights cannot be formed from empty strata")
        populated = [index for index, value in enumerate(counts) if value > 0.0]
        return cls(
            composite=composite,
            masks=tuple(
                tuple(task_masks[index] for index in populated) for task_masks in masks
            ),
            weights=np.asarray([counts[index] / total for index in populated], dtype=np.float64),
        )

    def risk(self, level: float) -> float:
        total = 0.0
        for stratum, weight in enumerate(self.weights):
            if weight <= 0.0:
                continue
            task_masks = tuple(task_masks[stratum] for task_masks in self.masks)
            total += float(weight) * self.composite.risk_with_masks(level, task_masks)
        return total

    def stratum_risk(self, level: float) -> FloatArray:
        values = np.zeros(self.weights.size, dtype=np.float64)
        for stratum in range(self.weights.size):
            task_masks = tuple(task_masks[stratum] for task_masks in self.masks)
            values[stratum] = self.composite.risk_with_masks(level, task_masks)
        return values

    def weight_error(self, other: FloatArray) -> float:
        if other.size != self.weights.size:
            raise ValueError("weight vectors must have equal length")
        return float(np.abs(self.weights - other).sum())

