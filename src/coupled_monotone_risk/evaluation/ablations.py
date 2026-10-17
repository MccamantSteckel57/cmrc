"""Ablations of the coupled construction (Sec. 4.4, Figure 3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from coupled_monotone_risk.calibration.cmrc import CmrcCalibrator
from coupled_monotone_risk.evaluation.protocol import ArmResult, ProtocolConfig, summarise
from coupled_monotone_risk.evaluation.runner import Prepared
from coupled_monotone_risk.risks.curve import RiskCurve


@dataclass(frozen=True)
class AblationOutcome:
    name: str
    arms: tuple[ArmResult, ...]
    worst_gap_pp: float
    allocation_residual: float


def _worst(arms: Sequence[ArmResult]) -> float:
    gaps = [arm.worst_gap_pp for arm in arms if np.isfinite(arm.worst_gap_pp)]
    return float(max(gaps)) if gaps else float("nan")


def independent_arm(
    prepared: Sequence[Prepared], config: ProtocolConfig, name: str = "independent"
) -> AblationOutcome:
    """Bare arm: one static calibrator per task at the level divided by the task count."""
    per_task = config.budget / len(prepared)
    arms: list[ArmResult] = []
    for entry in prepared:
        curve = RiskCurve.from_family(entry.task_input.family, config.grid_size)
        level = curve.generalized_inverse(per_task)
        arms.append(
            summarise(
                entry.task,
                arm=name,
                threshold=entry.task_input.family.threshold(level),
                level=level,
                budget=per_task,
            )
        )
    return AblationOutcome(name=name, arms=tuple(arms), worst_gap_pp=_worst(arms), allocation_residual=0.0)


def uniform_shared_arm(
    prepared: Sequence[Prepared], config: ProtocolConfig, name: str = "coupling_uniform"
) -> AblationOutcome:
    """Coupling restored on the bare arm with one share per task."""
    calibrator = CmrcCalibrator(
        budget=config.budget, grid_size=config.grid_size, strata=config.strata
    )
    shares = np.ones(len(prepared), dtype=np.float64)
    calibration = calibrator.fit(
        [entry.task_input for entry in prepared], shares=shares
    )
    arms = tuple(
        summarise(
            entry.task,
            arm=name,
            threshold=certificate.threshold,
            level=calibration.level,
            budget=config.budget,
        )
        for entry, certificate in zip(prepared, calibration.tasks, strict=True)
    )
    return AblationOutcome(
        name=name,
        arms=arms,
        worst_gap_pp=_worst(arms),
        allocation_residual=calibration.diagnostics.allocation_residual,
    )


def regime_blind_arm(
    prepared: Sequence[Prepared], config: ProtocolConfig, name: str = "pooled_strata"
) -> AblationOutcome:
    """Regime awareness removed: one pooled curve, no causal strata."""
    calibrator = CmrcCalibrator(
        budget=config.budget, grid_size=config.grid_size, strata=config.strata
    )
    calibration = calibrator.fit([entry.task_input for entry in prepared])
    arms = tuple(
        summarise(
            entry.task,
            arm=name,
            threshold=certificate.threshold,
            level=calibration.level,
            budget=config.budget,
        )
        for entry, certificate in zip(prepared, calibration.tasks, strict=True)
    )
    return AblationOutcome(
        name=name,
        arms=arms,
        worst_gap_pp=_worst(arms),
        allocation_residual=calibration.diagnostics.allocation_residual,
    )


def free_threshold_arm(
    prepared: Sequence[Prepared],
    config: ProtocolConfig,
    levels: int = 33,
    sweeps: int = 3,
    name: str = "free_thresholds",
) -> AblationOutcome:
    """Monotonicity removed: coordinate search over one free threshold per task.

    The objective is the weighted width subject to the pooled empirical risk on
    the calibration block. The result is not a function of one scalar, which is
    the condition Lemma 1 needs.
    """
    grid = np.linspace(0.0, 1.0, levels)
    weights = np.asarray([entry.task_input.weight for entry in prepared], dtype=np.float64)
    weights = weights / weights.sum()
    families = [entry.task_input.family for entry in prepared]
    chosen = np.full(len(prepared), 0.5, dtype=np.float64)

    def pooled(free: np.ndarray, attribute: str) -> float:
        values = np.asarray(
            [
                family.risk(float(level)) if attribute == "risk" else family.width(float(level))
                for family, level in zip(families, free, strict=True)
            ],
            dtype=np.float64,
        )
        return float(np.dot(weights, values))

    for _ in range(sweeps):
        for index in range(len(prepared)):
            best_value = np.inf
            best = float(chosen[index])
            for candidate in grid:
                trial = chosen.copy()
                trial[index] = candidate
                value = pooled(trial, "width") + 10.0 * max(
                    0.0, pooled(trial, "risk") - config.budget
                )
                if value < best_value:
                    best_value = value
                    best = float(candidate)
            chosen[index] = best
    arms = tuple(
        summarise(
            entry.task,
            arm=name,
            threshold=families[position].threshold(float(chosen[position])),
            level=float(chosen[position]),
            budget=config.budget,
        )
        for position, entry in enumerate(prepared)
    )
    surplus = max(0.0, pooled(chosen, "risk") - config.budget)
    return AblationOutcome(
        name=name, arms=arms, worst_gap_pp=_worst(arms), allocation_residual=float(surplus)
    )


def coupling_contribution(full: AblationOutcome, bare: AblationOutcome) -> float:
    return float(bare.worst_gap_pp - full.worst_gap_pp)


def restoration_share(full: AblationOutcome, bare: AblationOutcome, restored: AblationOutcome) -> float:
    total = bare.worst_gap_pp - full.worst_gap_pp
    if total <= 0.0:
        return float("nan")
    return float((bare.worst_gap_pp - restored.worst_gap_pp) / total)
