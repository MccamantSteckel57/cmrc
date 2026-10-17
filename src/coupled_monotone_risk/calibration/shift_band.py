"""The declared total-variation band across the calibration-to-test transition.

Ref: Remark 1, Sec. 3.4 (coverage >= 1 - alpha - sum_i w_i d_TV).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShiftBand:
    band: float
    bound: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.band < 1.0:
            raise ValueError("the band must lie inside [0, 1)")
        if self.bound <= 0.0:
            raise ValueError("the bound must be strictly positive")

    @property
    def propagation_error(self) -> float:
        return float(self.bound * self.band)

    def certificate_level(self, budget: float) -> float:
        if not 0.0 < budget < 1.0:
            raise ValueError("the budget must lie inside (0, 1)")
        level = budget + self.propagation_error
        if level >= 1.0:
            raise ValueError("the band pushes the certificate level to one")
        return float(level)

    def coverage_floor(self, nominal: float) -> float:
        return float(max(0.0, 1.0 - self.certificate_level(1.0 - nominal)))
