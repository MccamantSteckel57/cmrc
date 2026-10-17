"""Run every verification check and write the three top-level reports."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coupled_monotone_risk.cli.common import add_common, resolve_config
from coupled_monotone_risk.evaluation import verification
from coupled_monotone_risk.evaluation.verification import CheckResult
from coupled_monotone_risk.utils.atomic import atomic_write_json, sha256_file
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)

CHECKS = (
    verification.check_quantile_helpers,
    verification.check_isotonic,
    verification.check_interval_family,
    verification.check_set_family,
    verification.check_metrics_reference,
    verification.check_har_recovery,
    verification.check_baselines,
    verification.check_rank_correlation,
    verification.check_allocation_identity,
    verification.check_cmrc_coverage,
    verification.check_monotone_composite,
    verification.check_regime_stratification,
    verification.check_training_loop,
    verification.check_checkpoint,
    verification.check_shapes,
    verification.check_gradient,
    verification.check_determinism,
    verification.check_credit_pipeline,
    verification.check_dataset_availability,
    verification.check_volatility_layer,
    verification.check_regime_taxonomy,
    verification.check_regime_intervals,
    verification.check_coverage_interval_brackets,
    verification.check_figure1_claim,
    verification.check_discussion_gap,
    verification.check_width_claim,
    verification.check_algorithm_phases,
    verification.check_block_notation,
    verification.check_table_panels,
    verification.check_map_direction,
    verification.check_figure3_text,
    verification.check_panel_b_gaps,
    verification.check_panel_a_claims,
    verification.check_claim_table_alignment,
    verification.check_text_hygiene,
    verification.check_markdown_inventory,
    verification.check_empty_implementations,
    verification.check_duplicate_blocks,
    verification.check_module_inventory,
    verification.check_container_build,
)

CLAIMS: tuple[dict[str, Any], ...] = (
    {
        "id": "C01",
        "paper": {"section": "Sec. 3.2", "anchor": "Eq. (2)", "statement": "one shared scalar enters every task through a continuous strictly increasing coupling map"},
        "code": [
            {"path": "src/coupled_monotone_risk/coupling/maps.py", "symbol": "CouplingMap.__call__"},
            {"path": "src/coupled_monotone_risk/coupling/maps.py", "symbol": "CouplingMap.injective_share"},
        ],
        "checks": ["I03"],
        "manuscript_checks": ["M09"],
    },
    {
        "id": "C02",
        "paper": {"section": "Sec. 3.2", "anchor": "Eq. (3), Lemma 1", "statement": "the weighted composite is monotone on the shared axis"},
        "code": [
            {"path": "src/coupled_monotone_risk/coupling/composite.py", "symbol": "CompositeRisk.risk"},
            {"path": "src/coupled_monotone_risk/coupling/composite.py", "symbol": "CompositeRisk.non_monotone_share"},
        ],
        "checks": ["I03"],
        "manuscript_checks": [],
    },
    {
        "id": "C03",
        "paper": {"section": "Sec. 3.2", "anchor": "Eq. (4)-(6), Proposition 1", "statement": "budget shares equalise the marginal width cost and satisfy the budget identity"},
        "code": [
            {"path": "src/coupled_monotone_risk/coupling/allocation.py", "symbol": "equalised_marginal_allocation"},
            {"path": "src/coupled_monotone_risk/risks/cost.py", "symbol": "CostCurve.marginal_points"},
        ],
        "checks": ["I01"],
        "manuscript_checks": [],
    },
    {
        "id": "C04",
        "paper": {"section": "Sec. 3.3", "anchor": "Algorithm 1 lines 17-21", "statement": "the calibrated level is the lowest grid point meeting the shifted criterion"},
        "code": [
            {"path": "src/coupled_monotone_risk/coupling/composite.py", "symbol": "find_crossing"},
            {"path": "src/coupled_monotone_risk/calibration/cmrc.py", "symbol": "CmrcCalibrator.fit"},
        ],
        "checks": ["I02", "I04"],
        "manuscript_checks": [],
    },
    {
        "id": "C05",
        "paper": {"section": "Sec. 3.3", "anchor": "Algorithm 1 Phase A", "statement": "width-cost curves are fitted by isotonic regression and their residual is reported"},
        "code": [
            {"path": "src/coupled_monotone_risk/risks/isotonic.py", "symbol": "fit_decreasing"},
            {"path": "src/coupled_monotone_risk/risks/cost.py", "symbol": "CostCurve.from_curve"},
        ],
        "checks": ["U02", "I01"],
        "manuscript_checks": [],
    },
    {
        "id": "C06",
        "paper": {"section": "Remark 1", "anchor": "Algorithm 1 Phase D", "statement": "risk is averaged over causal strata with block weights"},
        "code": [
            {"path": "src/coupled_monotone_risk/coupling/composite.py", "symbol": "StratifiedView.from_labels"},
            {"path": "src/coupled_monotone_risk/data/strata.py", "symbol": "assign_strata"},
        ],
        "checks": ["I05"],
        "manuscript_checks": [],
    },
    {
        "id": "C07",
        "paper": {"section": "Remark 1", "anchor": "Sec. 3.4", "statement": "the calibration-to-test shift enters as one declared band"},
        "code": [
            {"path": "src/coupled_monotone_risk/calibration/shift_band.py", "symbol": "ShiftBand.certificate_level"},
        ],
        "checks": ["I05"],
        "manuscript_checks": [],
    },
    {
        "id": "C08",
        "paper": {"section": "Sec. 3.4", "anchor": "Theorem 1, Eq. (8)-(9)", "statement": "the joint finite-sample bound separates sampling and estimation slack"},
        "code": [
            {"path": "src/coupled_monotone_risk/bounds/finite_sample.py", "symbol": "theorem1_bound"},
            {"path": "src/coupled_monotone_risk/bounds/finite_sample.py", "symbol": "sampling_slack"},
        ],
        "checks": ["I02"],
        "manuscript_checks": [],
    },
    {
        "id": "C09",
        "paper": {"section": "Sec. 3.4", "anchor": "Eq. (10), Figure 5", "statement": "the budget-inflation relation brackets the exponent between zero and one"},
        "code": [
            {"path": "src/coupled_monotone_risk/bounds/inflation.py", "symbol": "inflation_bounds"},
            {"path": "src/coupled_monotone_risk/bounds/inflation.py", "symbol": "measured_exponent"},
        ],
        "checks": ["U06"],
        "manuscript_checks": [],
    },
    {
        "id": "C10",
        "paper": {"section": "Sec. 3.4", "anchor": "adaptive variant", "statement": "stratum weights follow an exponential update with a long-run statement"},
        "code": [
            {"path": "src/coupled_monotone_risk/calibration/adaptive.py", "symbol": "AdaptiveStratumWeights.update"},
            {"path": "src/coupled_monotone_risk/calibration/adaptive.py", "symbol": "drift_term"},
        ],
        "checks": ["I05"],
        "manuscript_checks": [],
    },
    {
        "id": "C11",
        "paper": {"section": "Sec. 4.1", "anchor": "regime taxonomy", "statement": "seven frozen windows label results after they are produced"},
        "code": [
            {"path": "src/coupled_monotone_risk/data/regimes.py", "symbol": "REGIME_WINDOWS"},
            {"path": "src/coupled_monotone_risk/data/regimes.py", "symbol": "label_for"},
        ],
        "checks": ["D01"],
        "manuscript_checks": ["M01", "M02"],
    },
    {
        "id": "C12",
        "paper": {"section": "Sec. 4.1", "anchor": "protocol", "statement": "blocks are chronological with an embargo and a rolling window"},
        "code": [
            {"path": "src/coupled_monotone_risk/data/splits.py", "symbol": "plan_blocks"},
            {"path": "src/coupled_monotone_risk/data/splits.py", "symbol": "rolling_windows"},
        ],
        "checks": ["I04"],
        "manuscript_checks": ["M07"],
    },
    {
        "id": "C13",
        "paper": {"section": "Sec. 4.2", "anchor": "HAR family", "statement": "daily, weekly and monthly realized-variance lags drive the point forecast"},
        "code": [
            {"path": "src/coupled_monotone_risk/data/features.py", "symbol": "har_design"},
            {"path": "src/coupled_monotone_risk/models/har.py", "symbol": "HarForecaster.fit"},
        ],
        "checks": ["U07"],
        "manuscript_checks": [],
    },
    {
        "id": "C14",
        "paper": {"section": "Sec. 4.2", "anchor": "backbones", "statement": "patch-transformer and selective state-space backbones supply point forecasts"},
        "code": [
            {"path": "src/coupled_monotone_risk/models/patchtst.py", "symbol": "PatchTST.forward"},
            {"path": "src/coupled_monotone_risk/models/mamba2.py", "symbol": "SelectiveStateSpaceBlock.forward"},
        ],
        "checks": ["S01", "G03"],
        "manuscript_checks": [],
    },
    {
        "id": "C15",
        "paper": {"section": "Sec. 4.2", "anchor": "metrics", "statement": "QLIKE, MSE, MDA, MAD, Winkler and CRPS are computed under one protocol"},
        "code": [
            {"path": "src/coupled_monotone_risk/metrics/point.py", "symbol": "qlike"},
            {"path": "src/coupled_monotone_risk/metrics/interval.py", "symbol": "winkler_score"},
        ],
        "checks": ["U05"],
        "manuscript_checks": [],
    },
    {
        "id": "C16",
        "paper": {"section": "Sec. 4.6", "anchor": "Table 4.6", "statement": "each regime is scored inside its own window with a binomial interval"},
        "code": [
            {"path": "src/coupled_monotone_risk/evaluation/protocol.py", "symbol": "summarise"},
            {"path": "src/coupled_monotone_risk/metrics/backtest.py", "symbol": "kupiec_pof"},
        ],
        "checks": ["I04"],
        "manuscript_checks": ["M02", "M13"],
    },
    {
        "id": "C17",
        "paper": {"section": "Sec. 4.4", "anchor": "Figure 3", "statement": "coupling, monotonicity and regime awareness are ablated separately"},
        "code": [
            {"path": "src/coupled_monotone_risk/evaluation/ablations.py", "symbol": "independent_arm"},
            {"path": "src/coupled_monotone_risk/evaluation/ablations.py", "symbol": "free_threshold_arm"},
        ],
        "checks": ["I04"],
        "manuscript_checks": [],
    },
    {
        "id": "C18",
        "paper": {"section": "Sec. 4.7", "anchor": "width claim", "statement": "the joint width is compared with independent calibrators at the same level"},
        "code": [
            {"path": "src/coupled_monotone_risk/evaluation/tradeoff.py", "symbol": "width_comparison"},
            {"path": "src/coupled_monotone_risk/evaluation/tradeoff.py", "symbol": "dispersion_ratio"},
        ],
        "checks": ["I04"],
        "manuscript_checks": ["M05"],
    },
    {
        "id": "C19",
        "paper": {"section": "Sec. 4.7", "anchor": "phase-alignment screen", "statement": "a residual rank correlation below 0.3 voids the joint-tightness statement"},
        "code": [
            {"path": "src/coupled_monotone_risk/bounds/inflation.py", "symbol": "screen_alignment"},
            {"path": "src/coupled_monotone_risk/bounds/inflation.py", "symbol": "residual_rank_correlation"},
        ],
        "checks": ["U06"],
        "manuscript_checks": [],
    },
    {
        "id": "C20",
        "paper": {"section": "Sec. 4.2", "anchor": "Panel B baselines", "statement": "adaptive and quantile baselines run under the same protocol"},
        "code": [
            {"path": "src/coupled_monotone_risk/calibration/baselines.py", "symbol": "OnlineConformal.run"},
            {"path": "src/coupled_monotone_risk/calibration/baselines.py", "symbol": "ConformalisedQuantile.fit"},
        ],
        "checks": ["U08"],
        "manuscript_checks": [],
    },
)


def _status_summary(checks: Sequence[CheckResult]) -> dict[str, int]:
    summary = {verification.PASS: 0, verification.FAIL: 0, verification.NOT_RUN: 0, verification.BLOCKED: 0}
    for check in checks:
        summary[check.status] = summary.get(check.status, 0) + 1
    return summary


def _code_status(checks: Sequence[CheckResult]) -> str:
    code_layers = {"unit", "integration", "training", "shape", "gradient", "regression", "hygiene", "data"}
    relevant = [check.status for check in checks if check.layer in code_layers]
    if verification.FAIL in relevant:
        return "FAIL"
    if verification.BLOCKED in relevant or verification.NOT_RUN in relevant:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def _claim_status(claim: dict[str, Any], checks: dict[str, CheckResult]) -> str:
    statuses = [checks[identifier].status for identifier in claim["checks"] if identifier in checks]
    if not statuses:
        return verification.NOT_RUN
    if verification.FAIL in statuses:
        return verification.FAIL
    if verification.BLOCKED in statuses:
        return verification.BLOCKED
    if verification.NOT_RUN in statuses:
        return verification.NOT_RUN
    return verification.PASS


def _environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _manifest(root: Path) -> dict[str, Any]:
    excluded_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "artefacts", ".git"}
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded_dirs for part in path.parts):
            continue
        if path.name == "integrity_manifest.json":
            continue
        relative = str(path.relative_to(root))
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        "".join(f"{entry['path']}:{entry['sha256']}" for entry in entries).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "file_count": len(entries),
        "root_digest": digest,
        "excluded": [*sorted(excluded_dirs), "integrity_manifest.json"],
        "files": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the verification suite")
    add_common(parser)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--skip-data", action="store_true")
    arguments = parser.parse_args(argv)
    resolve_config(arguments)
    root = arguments.root.resolve()

    checks: list[CheckResult] = []
    for factory in CHECKS:
        if arguments.skip_data and factory is verification.check_credit_pipeline:
            checks.append(
                CheckResult("I04", "integration", "credit panels train, calibrate and evaluate end to end", verification.NOT_RUN, {"reason": "skipped by flag"})
            )
            continue
        result = factory()
        LOGGER.info("%s %-7s %s", result.identifier, result.status, result.description)
        checks.append(result)

    by_identifier = {check.identifier: check for check in checks}
    summary = _status_summary(checks)
    code_status = _code_status(checks)
    manuscript_layers = [check.status for check in checks if check.layer == "manuscript"]
    manuscript_status = verification.FAIL if verification.FAIL in manuscript_layers else verification.PASS
    if code_status == "VERIFIED" and manuscript_status == verification.PASS:
        overall = "VERIFIED"
    elif code_status == "FAIL":
        overall = "UNVERIFIED"
    else:
        overall = "PARTIALLY_VERIFIED"

    claims_payload = {
        "project": "Distribution-Free Uncertainty Quantification for Structured Financial Prediction via Conformal Risk Control",
        "package": "coupled_monotone_risk",
        "claims": [
            {
                "id": claim["id"],
                "paper": claim["paper"],
                "code": claim["code"],
                "checks": claim["checks"],
                "manuscript_checks": claim["manuscript_checks"],
                "status": _claim_status(claim, by_identifier),
                "manuscript_status": _claim_status(
                    {"checks": claim["manuscript_checks"]}, by_identifier
                )
                if claim["manuscript_checks"]
                else None,
            }
            for claim in CLAIMS
        ],
    }

    report = {
        "project": "Distribution-Free Uncertainty Quantification for Structured Financial Prediction via Conformal Risk Control",
        "package": "coupled_monotone_risk",
        "overall_status": overall,
        "code_status": code_status,
        "manuscript_status": manuscript_status,
        "summary": summary,
        "environment": _environment(),
        "checks": [check.record() for check in checks],
        "notes": {
            "valuation": "results come only from executed code paths; manuscript findings are kept separate from code findings",
            "blocked_layers": [check.identifier for check in checks if check.status == verification.BLOCKED],
        },
    }

    atomic_write_json(root / "claim_to_code.json", claims_payload)
    atomic_write_json(root / "verification_report.json", report)
    atomic_write_json(root / "integrity_manifest.json", _manifest(root))
    LOGGER.info("overall %s code %s manuscript %s", overall, code_status, manuscript_status)
    return 0 if overall != "UNVERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())