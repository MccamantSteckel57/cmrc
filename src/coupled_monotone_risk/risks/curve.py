"""Empirical risk and width curves on the shared level axis (Sec. 3.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.risks.family import Mask, NestedFamily
from coupled_monotone_risk.utils.numeric import FloatArray, invert_decreasing


@dataclass(frozen=True)
class RiskCurve:
    levels: FloatArray
    risk: FloatArray
    width: FloatArray
    family: NestedFamily

    def __post_init__(self) -> None:
        if not (self.levels.size == self.risk.size == self.width.size):
            raise ValueError("risk curve arrays must have equal length")
        if np.any(np.diff(self.levels) <= 0.0):
            raise ValueError("levels must be strictly increasing")

    @classmethod
    def from_family(cls, family: NestedFamily, grid_size: int = 1001) -> RiskCurve:
        if grid_size < 3:
            raise ValueError("the level grid needs at least three points")
        levels = np.linspace(0.0, 1.0, grid_size)
        risk = np.asarray([family.risk(float(level)) for level in levels], dtype=np.float64)
        width = np.asarray([family.width(float(level)) for level in levels], dtype=np.float64)
        return cls(levels=levels, risk=risk, width=width, family=family)

    def generalized_inverse(self, target: float) -> float:
        """Lowest level whose empirical risk meets ``target`` with the rank correction.

        Requiring ``risk <= target - (1 - target) / n`` is the risk-axis form of
        the split-conformal rank ``ceil((n + 1) * (1 - target))``.
        """
        count = self.family.size
        corrected = float(target) - (1.0 - float(target)) / count
        return invert_decreasing(self.levels, self.risk, corrected)

    def risk_at(self, level: float) -> float:
        return float(np.interp(float(level), self.levels, self.risk))

    def width_at(self, level: float) -> float:
        return float(np.interp(float(level), self.levels, self.width))

    def monotone_share(self) -> float:
        steps = np.diff(self.risk)
        return float(np.mean(steps <= 1e-12))

    def lipschitz(self) -> float:
        steps = np.abs(np.diff(self.risk)) / np.diff(self.levels)
        return float(np.max(steps))

    def slope(self) -> float:
        """Mean absolute slope of the risk curve, used as ``gamma_k`` in Eq. (9)."""
        steps = np.abs(np.diff(self.risk))
        return float(np.sum(steps) / (self.levels[-1] - self.levels[0]))

    def right_continuous_at(self, level: float, tolerance: float = 1e-9) -> bool:
        if level >= 1.0:
            return True
        upper = float(min(1.0, level + tolerance))
        return abs(self.risk_at(level) - self.risk_at(upper)) <= 1e-6

    def quantile_threshold(self, level: float) -> float:
        return self.family.threshold(float(level))

    def masked_risk(self, mask: Mask, level: float) -> float:
        return self.family.risk(float(level), mask)
