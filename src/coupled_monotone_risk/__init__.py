"""Coupled monotone risk control for structured financial prediction."""

from __future__ import annotations

from coupled_monotone_risk.calibration.cmrc import (
    CalibrationResult,
    CmrcCalibrator,
    TaskCertificate,
    TaskInput,
)
from coupled_monotone_risk.coupling.allocation import (
    AllocationResult,
    equalised_marginal_allocation,
)
from coupled_monotone_risk.coupling.composite import CompositeRisk, find_crossing
from coupled_monotone_risk.coupling.maps import CouplingMap, build_maps
from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.risks.family import IntervalFamily, SetFamily

__version__ = "1.0.0"

__all__ = [
    "AllocationResult",
    "CalibrationResult",
    "CmrcCalibrator",
    "CompositeRisk",
    "CostCurve",
    "CouplingMap",
    "IntervalFamily",
    "RiskCurve",
    "SetFamily",
    "TaskCertificate",
    "TaskInput",
    "build_maps",
    "equalised_marginal_allocation",
    "find_crossing",
]
