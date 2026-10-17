"""Theorem 1: the joint finite-sample bound on the composite risk.

Ref: Theorem 1, Eq. (8)-(9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.utils.numeric import FloatArray


@dataclass(frozen=True)
class BoundInputs:
    sample_size: int
    task_count: int
    grid_size: int
    budget: float
    delta: float
    bound: float
    weights: FloatArray
    lipschitz: FloatArray
    slope: FloatArray
    share_errors: FloatArray
    weight_error: float

    def __post_init__(self) -> None:
        if self.sample_size < 2:
            raise ValueError("the bound needs a calibration block of at least two")
        if self.task_count != self.weights.size:
            raise ValueError("one weight per task is required")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie inside (0, 1)")
        if self.grid_size < 2:
            raise ValueError("the level grid needs at least two points")
        if np.any(self.slope <= 0.0):
            raise ValueError("the slope of every task risk curve must be positive")


@dataclass(frozen=True)
class BoundReport:
    sampling_slack: float
    estimation_slack: float
    total_slack: float
    per_task_slack: FloatArray
    delta: float
    budget: float
    certificate: float

    def exceeds_budget(self, observed_risk: float) -> bool:
        return bool(observed_risk > self.certificate)


def sampling_slack(inputs: BoundInputs) -> float:
    numerator = (
        math.log(inputs.grid_size)
        + math.log(1.0 / inputs.delta)
        + math.log(inputs.task_count)
    )
    return float(inputs.bound * math.sqrt(numerator / (2.0 * inputs.sample_size)))


def per_task_slack(inputs: BoundInputs) -> FloatArray:
    numerator = math.log(inputs.grid_size) + math.log(inputs.task_count / inputs.delta)
    base = float(inputs.bound * math.sqrt(numerator / (2.0 * inputs.sample_size)))
    return np.full(inputs.task_count, base, dtype=np.float64)


def estimation_slack(inputs: BoundInputs) -> float:
    task_terms = (
        inputs.weights
        * (inputs.lipschitz / inputs.slope)
        * (per_task_slack(inputs) + inputs.budget * np.abs(inputs.share_errors))
    )
    return float(task_terms.sum() + inputs.bound * inputs.weight_error)


def theorem1_bound(inputs: BoundInputs) -> BoundReport:
    sampling = sampling_slack(inputs)
    estimation = estimation_slack(inputs)
    total = sampling + estimation
    return BoundReport(
        sampling_slack=sampling,
        estimation_slack=estimation,
        total_slack=total,
        per_task_slack=per_task_slack(inputs),
        delta=inputs.delta,
        budget=inputs.budget,
        certificate=float(inputs.budget + total),
    )
