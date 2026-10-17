"""Calibrate the coupled construction on one composite study."""

from __future__ import annotations

import argparse

from coupled_monotone_risk.cli.common import add_common, emit, resolve_config
from coupled_monotone_risk.data.credit import load_german, load_taiwan
from coupled_monotone_risk.data.manifest import MissingDataError
from coupled_monotone_risk.evaluation.ablations import independent_arm, uniform_shared_arm
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig
from coupled_monotone_risk.evaluation.reporting import (
    adaptive_panel,
    certificate_panel,
    credit_panel_metrics,
    inflation_panel,
    online_baseline_panel,
)
from coupled_monotone_risk.evaluation.runner import prepare_credit_arm, run_study
from coupled_monotone_risk.evaluation.tables import main_rows, regime_rows
from coupled_monotone_risk.evaluation.tradeoff import width_comparison
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOADERS = {"german": load_german, "taiwan": load_taiwan}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="calibrate a composite credit study")
    add_common(parser)
    parser.add_argument("--panels", nargs="+", default=["german", "taiwan"])
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--grid", type=int, default=1001)
    parser.add_argument("--ablations", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = resolve_config(arguments)
    budget = float(config.get("protocol", {}).get("budget", arguments.budget))
    protocol = ProtocolConfig(
        budget=budget, epochs=arguments.epochs, seed=arguments.seed, grid_size=arguments.grid
    )
    prepared = []
    reports = []
    for name in arguments.panels:
        if name not in LOADERS:
            LOGGER.error("unknown panel %s", name)
            return 2
        try:
            panel = LOADERS[name]()
        except (MissingDataError, FileNotFoundError) as error:
            LOGGER.error("panel unavailable: %s", error)
            return 3
        entry, report = prepare_credit_arm(
            panel, "credit_mlp", protocol, weight=1.0 / len(arguments.panels)
        )
        prepared.append(entry)
        reports.append(report)
    study = run_study(prepared, protocol, arm="cmrc", reports=reports)
    comparison = width_comparison(tuple(prepared), study.calibration, protocol)
    payload: dict[str, object] = {
        "panels": list(arguments.panels),
        "budget": budget,
        "level": round(study.calibration.level, 6),
        "achieved_risk": round(study.calibration.achieved_risk, 6),
        "shifted_risk": round(study.calibration.shifted_risk, 6),
        "shares": {task.name: round(task.share, 6) for task in study.calibration.tasks},
        "budgets": {task.name: round(task.budget, 6) for task in study.calibration.tasks},
        "widths": {task.name: round(task.width, 6) for task in study.calibration.tasks},
        "cost_residuals": [
            round(float(value), 8) for value in study.calibration.diagnostics.cost_residuals
        ],
        "allocation_residual": round(study.calibration.diagnostics.allocation_residual, 8),
        "width_ratio_vs_independent": round(comparison.ratio_vs_independent, 6),
        "width_ratio_vs_uniform": round(comparison.ratio_vs_uniform, 6),
        "rows": main_rows(study.arms),
        "cells": regime_rows(study.arms),
        "certificate": certificate_panel(tuple(prepared), study.calibration, protocol),
        "inflation": inflation_panel(tuple(prepared), study.calibration),
        "adaptive": adaptive_panel(study.calibration),
        "discrimination": credit_panel_metrics(tuple(prepared), study.calibration.thresholds()),
        "baselines": online_baseline_panel(tuple(prepared), budget),
    }
    if arguments.ablations:
        bare = independent_arm(tuple(prepared), protocol)
        restored = uniform_shared_arm(tuple(prepared), protocol)
        payload["ablations"] = {
            "bare_worst_gap_pp": round(bare.worst_gap_pp, 4),
            "coupling_worst_gap_pp": round(restored.worst_gap_pp, 4),
            "full_worst_gap_pp": round(max(arm.worst_gap_pp for arm in study.arms), 4),
        }
    emit(payload, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
