"""Evaluate every arm whose declared source is present, and record the rest."""

from __future__ import annotations

import argparse

import numpy as np

from coupled_monotone_risk.cli.common import add_common, emit, resolve_config
from coupled_monotone_risk.data.credit import load_available
from coupled_monotone_risk.data.equity import load_pair
from coupled_monotone_risk.data.manifest import MissingDataError, availability
from coupled_monotone_risk.data.omrl import load_subset
from coupled_monotone_risk.evaluation.ablations import independent_arm
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig
from coupled_monotone_risk.evaluation.reporting import (
    credit_panel_metrics,
    online_baseline_panel,
    ranking_panel,
)
from coupled_monotone_risk.evaluation.runner import (
    prepare_credit_arm,
    prepare_return_arm,
    prepare_volatility_arm,
    run_study,
)
from coupled_monotone_risk.evaluation.tables import main_rows, regime_rows
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run the evaluation grid")
    add_common(parser)
    parser.add_argument("--layer", choices=("credit", "volatility", "returns", "all"), default="all")
    parser.add_argument("--models", nargs="+", default=["har", "har_log"])
    parser.add_argument("--return-models", nargs="+", default=["ar"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    resolve_config(arguments)
    protocol = ProtocolConfig(epochs=arguments.epochs, seed=arguments.seed)
    blocked: list[dict[str, object]] = []
    payload: dict[str, object] = {"runs": [], "blocked": blocked, "availability": availability()}
    runs: list[dict[str, object]] = []

    if arguments.layer in ("credit", "all"):
        panels = load_available()
        if not panels:
            blocked.append({"layer": "credit", "reason": "no credit panel present"})
        else:
            prepared = []
            reports = []
            share = 1.0 / len(panels)
            for panel in panels.values():
                entry, report = prepare_credit_arm(panel, "credit_mlp", protocol, weight=share)
                prepared.append(entry)
                if report is not None:
                    reports.append(report)
            if prepared:
                study = run_study(prepared, protocol, arm="cmrc", reports=reports)
                bare = independent_arm(tuple(prepared), protocol)
                calibrated_losses: dict[str, np.ndarray] = {}
                bare_losses: dict[str, np.ndarray] = {}
                share = protocol.budget / len(prepared)
                for entry, certificate in zip(prepared, study.calibration.tasks, strict=True):
                    curve = RiskCurve.from_family(entry.task_input.family, protocol.grid_size)
                    bare_threshold = entry.task_input.family.threshold(
                        curve.generalized_inverse(share)
                    )
                    name = entry.task.name
                    calibrated_losses[name] = 1.0 - entry.task.hits(certificate.threshold)
                    bare_losses[name] = 1.0 - entry.task.hits(bare_threshold)
                runs.append(
                    {
                        "layer": "credit",
                        "arms": main_rows(study.arms),
                        "cells": regime_rows(study.arms),
                        "discrimination": credit_panel_metrics(
                            tuple(prepared), study.calibration.thresholds()
                        ),
                        "baselines": online_baseline_panel(tuple(prepared), protocol.budget),
                        "bare_worst_gap_pp": round(bare.worst_gap_pp, 4),
                        "ranking": ranking_panel(calibrated_losses, bare_losses),
                    }
                )

    if arguments.layer in ("volatility", "all"):
        series = load_subset()
        if not series:
            blocked.append(
                {"layer": "volatility", "reason": "realized-library archive not present"}
            )
        else:
            for model_name in arguments.models:
                prepared = []
                share = 1.0 / len(series)
                for asset in series.values():
                    try:
                        entry, _ = prepare_volatility_arm(asset, model_name, protocol, weight=share)
                    except (ValueError, MissingDataError) as error:
                        LOGGER.info("skipping %s: %s", asset.symbol, error)
                        continue
                    prepared.append(entry)
                if prepared:
                    study = run_study(prepared, protocol, arm=f"cmrc_{model_name}")
                    runs.append(
                        {
                            "layer": "volatility",
                            "model": model_name,
                            "arms": main_rows(study.arms),
                            "cells": regime_rows(study.arms),
                        }
                    )

    if arguments.layer in ("returns", "all"):
        pair = load_pair()
        if not pair:
            blocked.append({"layer": "returns", "reason": "equity series not present"})
        else:
            for model_name in arguments.return_models:
                prepared = []
                share = 1.0 / len(pair)
                for asset in pair.values():
                    entry, _ = prepare_return_arm(asset, model_name, protocol, weight=share)
                    prepared.append(entry)
                if prepared:
                    study = run_study(prepared, protocol, arm=f"cmrc_{model_name}")
                    runs.append(
                        {
                            "layer": "returns",
                            "model": model_name,
                            "arms": main_rows(study.arms),
                            "cells": regime_rows(study.arms),
                        }
                    )

    payload["runs"] = runs
    emit(payload, arguments.output)
    return 0 if runs else 3


if __name__ == "__main__":
    raise SystemExit(main())
