"""Width, dispersion and timing of the joint calibration (Sec. 4.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.calibration.cmrc import CalibrationResult
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig
from coupled_monotone_risk.evaluation.runner import Prepared
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.utils.seeding import set_seed


@dataclass(frozen=True)
class WidthComparison:
    joint_total: float
    independent_total: float
    uniform_total: float
    allocated_total: float

    @property
    def ratio_vs_independent(self) -> float:
        if self.independent_total <= 0.0:
            return float("nan")
        return float(self.joint_total / self.independent_total)

    @property
    def ratio_vs_uniform(self) -> float:
        if self.uniform_total <= 0.0:
            return float("nan")
        return float(self.allocated_total / self.uniform_total)


def width_comparison(
    prepared: tuple[Prepared, ...], calibration: CalibrationResult, config: ProtocolConfig
) -> WidthComparison:
    joint = float(sum(task.width for task in calibration.tasks))
    independent = 0.0
    for entry in prepared:
        curve = RiskCurve.from_family(entry.task_input.family, config.grid_size)
        level = curve.generalized_inverse(config.budget)
        independent += float(entry.task_input.family.width(level))
    return WidthComparison(
        joint_total=joint,
        independent_total=float(independent),
        uniform_total=float(calibration.diagnostics.uniform_width),
        allocated_total=float(calibration.diagnostics.allocated_width),
    )


def dispersion_ratio(
    scores: tuple[np.ndarray, ...],
    weights: np.ndarray,
    thresholds: tuple[float, ...],
    budget: float,
    replicates: int = 200,
    seed: int = 11,
) -> float:
    """Standard deviation of the realised composite risk over resampled blocks."""
    set_seed(seed)
    generator = np.random.default_rng(seed)
    realised = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        contributions = np.asarray(
            [
                float(np.mean(values[generator.integers(0, values.size, size=values.size)] > threshold))
                for values, threshold in zip(scores, thresholds, strict=True)
            ],
            dtype=np.float64,
        )
        realised[draw] = float(np.dot(weights, contributions))
    spread = float(np.std(realised, ddof=1))
    return spread / budget if budget > 0.0 else float("nan")
