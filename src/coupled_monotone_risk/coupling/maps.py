"""Coupling maps from the shared scalar onto each task family (Eq. (2)-(3))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.utils.numeric import FloatArray

LEVEL_GRID_SIZE = 257


@dataclass(frozen=True)
class CouplingMap:
    task: str
    curve: RiskCurve
    share: float
    budget: float

    def __post_init__(self) -> None:
        if not self.share > 0.0:
            raise ValueError("coupling shares must be strictly positive")
        if not 0.0 < self.budget < 1.0:
            raise ValueError("the aggregate budget must lie inside (0, 1)")

    def __call__(self, level: float) -> float:
        """Per-task level at shared coordinate ``level``.

        ``level`` is the shrinkage away from the allocated budget, so ``0`` sits
        on the calibrated budget ``c_k * alpha`` and ``1`` opens the family to its
        widest member. Eq. (2) writes the same map as ``G^-1(c_k * lambda * alpha)``
        with the complementary coordinate; the object at the calibration point is
        identical and the composite is decreasing in the coordinate used here.
        """
        budget_term = self.share * self.budget * (1.0 - float(level))
        return float(self.curve.generalized_inverse(budget_term))

    @property
    def task_budget(self) -> float:
        return self.share * self.budget

    def task_risk(self, level: float, mask: object = None) -> float:
        target = self.__call__(level)
        if mask is None:
            return float(self.curve.family.risk(target))
        return float(self.curve.family.risk(target, mask))  # type: ignore[arg-type]

    def calibrated_level(self) -> float:
        return self.__call__(0.0)

    def widest_level(self) -> float:
        return self.__call__(1.0)

    def grid(self) -> tuple[FloatArray, FloatArray]:
        levels = np.linspace(0.0, 1.0, LEVEL_GRID_SIZE)
        mapped = np.asarray([self(level) for level in levels], dtype=np.float64)
        return levels, mapped

    def monotone_profile(self) -> tuple[float, float]:
        _, mapped = self.grid()
        increments = np.diff(mapped)
        share_positive = float(np.mean(increments >= 0.0))
        share_strict = float(np.mean(increments > 1e-12))
        return share_positive, share_strict

    def injective_share(self) -> float:
        """Share of distinct images on the non-saturated part of the grid."""
        _, mapped = self.grid()
        active = mapped < mapped[-1] - 1e-12
        if not np.any(active):
            return 1.0
        return float(np.unique(np.round(mapped[active], 12)).size / int(active.sum()))

    def saturation_share(self) -> float:
        _, mapped = self.grid()
        return float(np.mean(mapped >= mapped[-1] - 1e-12))


def build_maps(
    tasks: tuple[str, ...],
    curves: tuple[RiskCurve, ...],
    shares: FloatArray,
    budget: float,
) -> tuple[CouplingMap, ...]:
    if not (len(tasks) == len(curves) == shares.size):
        raise ValueError("tasks, curves and shares must be aligned")
    return tuple(
        CouplingMap(task=name, curve=curve, share=float(share), budget=budget)
        for name, curve, share in zip(tasks, curves, shares, strict=True)
    )
