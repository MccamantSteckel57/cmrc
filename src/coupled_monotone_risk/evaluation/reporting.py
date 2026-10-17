"""Study-level panels: bound certificate, inflation bounds, adaptive weights, ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from coupled_monotone_risk.bounds.finite_sample import BoundInputs, theorem1_bound
from coupled_monotone_risk.bounds.inflation import (
    VOID_THRESHOLD,
    inflation_bounds,
    residual_rank_correlation,
    screen_alignment,
)
from coupled_monotone_risk.calibration.adaptive import AdaptiveStratumWeights, drift_term
from coupled_monotone_risk.calibration.cmrc import CalibrationResult
from coupled_monotone_risk.calibration.shift_band import ShiftBand
from coupled_monotone_risk.coupling.allocation import equalised_marginal_allocation
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig, SetTask
from coupled_monotone_risk.evaluation.runner import Prepared
from coupled_monotone_risk.metrics.classification import (
    auc_roc,
    average_precision,
    brier_score,
    expected_calibration_error,
    ks_statistic,
)
from coupled_monotone_risk.metrics.ranking import diebold_mariano
from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.risks.curve import RiskCurve


def certificate_panel(
    prepared: Sequence[Prepared], calibration: CalibrationResult, config: ProtocolConfig
) -> dict[str, object]:
    families = [entry.task_input.family for entry in prepared]
    weights = np.asarray([entry.task_input.weight for entry in prepared], dtype=np.float64)
    weights = weights / weights.sum()
    half_shares = []
    for offset in (0, 1):
        halves = []
        for family in families:
            size = family.size
            mask = np.zeros(size, dtype=bool)
            mask[offset::2] = True
            halves.append(family.subset(mask))
        half_curves = tuple(
            CostCurve.from_curve(RiskCurve.from_family(family, config.grid_size)) for family in halves
        )
        half_shares.append(equalised_marginal_allocation(half_curves, weights, config.budget).shares)
    share_errors = np.abs(half_shares[0] - half_shares[1])
    risk_curves = [RiskCurve.from_family(family, config.grid_size) for family in families]
    inputs = BoundInputs(
        sample_size=int(calibration.diagnostics.effective_size),
        task_count=len(families),
        grid_size=config.grid_size,
        budget=config.budget,
        delta=config.delta,
        bound=1.0,
        weights=weights,
        lipschitz=np.asarray([curve.lipschitz() for curve in risk_curves], dtype=np.float64),
        slope=np.asarray([max(curve.slope(), 1e-6) for curve in risk_curves], dtype=np.float64),
        share_errors=share_errors,
        weight_error=0.0,
    )
    report = theorem1_bound(inputs)
    band = ShiftBand(band=calibration.diagnostics.shift_band, bound=1.0)
    return {
        "sampling_slack": round(report.sampling_slack, 6),
        "estimation_slack": round(report.estimation_slack, 6),
        "total_slack": round(report.total_slack, 6),
        "certificate_level": round(report.certificate, 6),
        "share_error": [round(float(value), 6) for value in share_errors],
        "band_certificate": round(band.certificate_level(config.budget), 6),
        "delta": config.delta,
    }


def inflation_panel(prepared: Sequence[Prepared], calibration: CalibrationResult) -> dict[str, object]:
    counts = np.arange(1, len(prepared) + 1, dtype=np.float64)
    bounds = inflation_bounds(counts, calibration.budget)
    length = min(entry.task_input.family.size for entry in prepared)
    scores = np.column_stack(
        [np.asarray(entry.task_input.family.scores[:length], dtype=np.float64) for entry in prepared]
    )
    clock = np.arange(length, dtype=np.float64)
    correlation = residual_rank_correlation(scores, clock) if scores.shape[1] > 1 else float("nan")
    screen = screen_alignment(correlation)
    return {
        "shared_budget": [round(float(value), 6) for value in bounds.shared],
        "simultaneous_budget": [round(float(value), 6) for value in bounds.simultaneous],
        "exponent_shared": bounds.exponent_shared,
        "exponent_simultaneous": bounds.exponent_simultaneous,
        "measured_exponent": None,
        "residual_rank_correlation": None if np.isnan(correlation) else round(correlation, 6),
        "void_threshold": VOID_THRESHOLD,
        "screen": screen.verdict,
        "note": "no exponent is fitted: Figure 5 plots only the two proved bounds",
    }


def adaptive_panel(calibrated: CalibrationResult, learning_rate: float = 0.5) -> dict[str, object]:
    strata_losses = np.asarray(calibrated.diagnostics.stratum_losses, dtype=np.float64)
    if strata_losses.size < 2:
        return {"status": "NOT_RUN", "reason": "no causal strata were supplied"}
    weights = AdaptiveStratumWeights(
        np.full(strata_losses.size, 1.0 / strata_losses.size), learning_rate
    )
    history = [weights.weights]
    for _ in range(5):
        history.append(weights.update(strata_losses))
    return {
        "status": "RUN",
        "learning_rate": learning_rate,
        "initial": [round(float(value), 6) for value in history[0]],
        "final": [round(float(value), 6) for value in history[-1]],
        "drift": round(drift_term(tuple(history)), 8),
        "long_run_allowance": round(drift_term(tuple(history)) / learning_rate, 8),
    }


def credit_panel_metrics(prepared: Sequence[Prepared], thresholds: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in prepared:
        task = entry.task
        if not isinstance(task, SetTask):
            continue
        probability = task.probabilities[:, 1]
        threshold = thresholds[task.name]
        included = task.probabilities >= (1.0 - threshold)
        rows.append(
            {
                "panel": task.name,
                "rows": int(task.truth.size),
                "auc": round(auc_roc(task.truth, probability), 6),
                "average_precision": round(average_precision(task.truth, probability), 6),
                "ks": round(ks_statistic(task.truth, probability), 6),
                "brier": round(brier_score(task.truth, probability), 6),
                "ece": round(expected_calibration_error(task.truth, probability), 6),
                "set_size": round(float(np.mean(np.maximum(included.sum(axis=1), 1))), 6),
                "coverage": round(float(task.hits(threshold).mean()), 6),
            }
        )
    return rows


def ranking_panel(
    primary: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, losses in primary.items():
        other = reference.get(name)
        if other is None or other.size != losses.size:
            continue
        outcome = diebold_mariano(losses, other)
        rows.append(
            {
                "task": name,
                "mean_loss_difference": round(outcome.mean_difference, 8),
                "statistic": round(outcome.statistic, 6),
                "p_value": round(outcome.p_value, 6),
                "lag": outcome.lag,
            }
        )
    return rows


def online_baseline_panel(
    prepared: Sequence[Prepared], budget: float, eta: float = 0.01
) -> list[dict[str, object]]:
    from coupled_monotone_risk.calibration.baselines import (
        OnlineConformal,
        split_conformal_threshold,
    )

    rows: list[dict[str, object]] = []
    for entry in prepared:
        scores = np.asarray(entry.task_input.family.scores, dtype=np.float64)
        if scores.size < 60:
            continue
        online = OnlineConformal(name="aci", eta=eta, window=min(250, scores.size - 1))
        result = online.run(scores, budget)
        static = split_conformal_threshold(scores, budget)
        rows.append(
            {
                "task": entry.task_input.name,
                "static_threshold": round(float(static), 6),
                "static_risk": round(float(np.mean(scores > static)), 6),
                "aci_coverage": round(result.mean_coverage, 6),
                "aci_final_level": round(result.final_level, 6),
                "eta": eta,
            }
        )
    return rows
