"""Executed verification checks for the release.

Every check returns the status it actually observed. Data-dependent checks are
BLOCKED when the declared source is absent, never silently passed.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from coupled_monotone_risk.calibration.cmrc import CmrcCalibrator, TaskInput
from coupled_monotone_risk.coupling.allocation import equalised_marginal_allocation
from coupled_monotone_risk.data.manifest import availability
from coupled_monotone_risk.data.regimes import PUBLICATION_COUNTS
from coupled_monotone_risk.evaluation.ablations import independent_arm, uniform_shared_arm
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig
from coupled_monotone_risk.evaluation.runner import prepare_credit_arm, run_study
from coupled_monotone_risk.metrics.backtest import kupiec_pof
from coupled_monotone_risk.metrics.interval import winkler_score
from coupled_monotone_risk.metrics.point import qlike
from coupled_monotone_risk.models.credit import CreditModel
from coupled_monotone_risk.models.mamba2 import MambaForecaster, SelectiveStateSpaceBlock
from coupled_monotone_risk.models.patchtst import PatchTST
from coupled_monotone_risk.models.recurrent import ReturnTransformer
from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.risks.family import IntervalFamily, SetFamily
from coupled_monotone_risk.risks.isotonic import fit_decreasing
from coupled_monotone_risk.training.ckpt import CheckpointMeta, load_checkpoint, save_checkpoint
from coupled_monotone_risk.training.trainer import TrainConfig
from coupled_monotone_risk.utils.numeric import conformal_quantile, empirical_quantile, spearman

PASS, FAIL, NOT_RUN, BLOCKED = "PASS", "FAIL", "NOT_RUN", "BLOCKED"


@dataclass(frozen=True)
class CheckResult:
    identifier: str
    layer: str
    description: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "layer": self.layer,
            "description": self.description,
            "status": self.status,
            "evidence": self.evidence,
        }


def _guarded(identifier: str, layer: str, description: str, body: Callable[[], tuple[str, dict[str, Any]]]) -> CheckResult:
    try:
        status, evidence = body()
    except Exception as error:
        return CheckResult(identifier, layer, description, FAIL, {"error": type(error).__name__, "detail": str(error)[:400]})
    return CheckResult(identifier, layer, description, status, evidence)


def check_quantile_helpers() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        sample = np.arange(1.0, 101.0)
        q50 = empirical_quantile(sample, 0.5)
        conformal = conformal_quantile(sample, 0.9)
        rank = math.ceil(101 * 0.9)
        ok = abs(q50 - 50.0) <= 1.0 and conformal == float(sample[rank - 1])
        return (PASS if ok else FAIL), {"median": q50, "conformal": conformal, "rank": rank}

    return _guarded("U01", "unit", "empirical and conformal quantiles match their definitions", body)


def check_isotonic() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        values = np.asarray([5.0, 4.0, 6.0, 3.0, 7.0, 2.0])
        fitted = fit_decreasing(np.arange(values.size, dtype=np.float64), values)
        monotone = bool(np.all(np.diff(fitted.fitted) <= 1e-12))
        mean_preserved = abs(float(fitted.fitted.mean()) - float(values.mean())) < 1e-9
        return (PASS if monotone and mean_preserved else FAIL), {
            "fitted": [round(float(value), 6) for value in fitted.fitted],
            "residual": round(fitted.residual, 6),
        }

    return _guarded("U02", "unit", "pool-adjacent-violators returns a decreasing fit with the sample mean", body)


def check_interval_family() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(4)
        family = IntervalFamily(np.abs(rng.normal(size=2000)), 1.0)
        risks = [family.risk(level) for level in (0.1, 0.5, 0.9)]
        decreasing = risks[0] >= risks[1] >= risks[2]
        right = abs(family.risk(0.37) - family.risk(0.370001)) < 1e-12
        return (PASS if decreasing and right else FAIL), {"risks": [round(value, 6) for value in risks]}

    return _guarded("U03", "unit", "interval family risk is non-increasing and right-continuous", body)


def check_set_family() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(5)
        probabilities = rng.dirichlet([1.0, 1.0], size=1500)
        labels = rng.integers(0, 2, size=1500)
        family = SetFamily(probabilities, labels)
        sizes = [family.width(level) for level in (0.1, 0.5, 0.95)]
        growing = sizes[0] <= sizes[1] <= sizes[2]
        monotone_risk = family.risk(0.2) >= family.risk(0.8)
        return (PASS if growing and monotone_risk else FAIL), {
            "sizes": [round(value, 6) for value in sizes]
        }

    return _guarded("U04", "unit", "set family size falls and risk is non-increasing in the level", body)


def check_allocation_identity() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(6)
        families = (
            IntervalFamily(np.abs(rng.normal(0.0, 1.0, 900)), 1.0),
            IntervalFamily(np.abs(rng.normal(0.0, 2.5, 700)), 2.0),
            SetFamily(rng.dirichlet([1.0, 1.0], size=800), rng.integers(0, 2, 800)),
        )
        weights = np.asarray([0.4, 0.4, 0.2])
        curves = tuple(CostCurve.from_curve(RiskCurve.from_family(f, 501)) for f in families)
        result = equalised_marginal_allocation(curves, weights, 0.1)
        identity = abs(float(np.dot(weights, result.shares)) - 1.0)
        margins = np.asarray([curve.marginal(value) for curve, value in zip(curves, result.budgets, strict=True)])
        spread = float(np.max(margins) - np.min(margins)) / max(float(np.mean(margins)), 1e-12)
        better = result.allocated_objective <= result.uniform_objective + 1e-9
        return (PASS if identity < 1e-9 and better else FAIL), {
            "share_identity_error": identity,
            "marginal_spread": round(spread, 6),
            "allocated": round(result.allocated_objective, 6),
            "uniform": round(result.uniform_objective, 6),
        }

    return _guarded("I01", "integration", "equalised allocation satisfies sum_k w_k c_k = 1 and beats the uniform split", body)


def check_cmrc_coverage() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(21)
        scales = (1.0, 1.7, 0.6)
        families = tuple(
            IntervalFamily(np.abs(rng.normal(0.0, scale, 900)), scale) for scale in scales
        )
        calibrator = CmrcCalibrator(budget=0.1, grid_size=1001)
        result = calibrator.fit(
            [TaskInput(f"task{index}", family, 1.0 / 3.0) for index, family in enumerate(families)]
        )
        coverage = []
        for index, scale in enumerate(scales):
            truth = np.abs(rng.normal(0.0, scale, 20000))
            threshold = result.tasks[index].threshold
            coverage.append(float(np.mean(truth <= threshold)))
        weights = np.full(len(scales), 1.0 / len(scales))
        composite_risk = float(np.dot(weights, 1.0 - np.asarray(coverage)))
        gap = abs(composite_risk - 0.1)
        return (PASS if gap <= 0.02 else FAIL), {
            "per_task_coverage": [round(value, 5) for value in coverage],
            "composite_risk": round(composite_risk, 5),
            "gap": round(gap, 5),
            "target": 0.1,
        }

    return _guarded("I02", "integration", "calibrated thresholds attain the nominal level on a fresh draw", body)


def check_monotone_composite() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(22)
        families = (
            IntervalFamily(np.abs(rng.normal(size=500)), 1.0),
            SetFamily(rng.dirichlet([1.0, 1.0], size=500), rng.integers(0, 2, 500)),
        )
        calibrator = CmrcCalibrator(budget=0.1, grid_size=401)
        result = calibrator.fit([TaskInput("a", families[0], 0.5), TaskInput("b", families[1], 0.5)])
        grid = np.linspace(0.0, 1.0, 41)
        values = np.asarray([result.composite.risk(float(level)) for level in grid])
        share = float(np.mean(np.diff(values) <= 1e-12))
        return (PASS if share == 1.0 else FAIL), {"non_increasing_share": share, "start": round(float(values[0]), 5), "end": round(float(values[-1]), 5)}

    return _guarded("I03", "integration", "composite loss falls along the shared axis", body)


def check_metrics_reference() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        actual = np.asarray([0.04, 0.09, 0.01, 0.16])
        perfect = qlike(actual, actual)
        coverage_hit = winkler_score(actual, actual, actual, 0.1)
        kupiec = kupiec_pof(10, 100, 0.1)
        ok = abs(perfect) < 1e-12 and abs(coverage_hit) < 1e-12 and kupiec.p_value > 0.9
        return (PASS if ok else FAIL), {
            "qlike_perfect": perfect,
            "winkler_perfect": coverage_hit,
            "kupiec_p": round(kupiec.p_value, 6),
        }

    return _guarded("U05", "unit", "point and interval metrics are zero for a perfect forecast", body)


def check_training_loop() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        features = rng.normal(size=(32, 6))
        labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(np.float64)
        model = CreditModel(width=6, config=TrainConfig(epochs=60, batch_size=16, lr=5e-3, seed=3, log_every=0), hidden=32)
        report = model.fit(features, labels)
        first = report.first_train_loss
        final = report.final_train_loss
        return (PASS if final < 0.5 * first and report.parameter_delta > 0.0 else FAIL), {
            "first_loss": round(first, 6),
            "final_loss": round(final, 6),
            "parameter_delta": round(report.parameter_delta, 6),
        }

    return _guarded("G01", "training", "a separable batch drives the loss down and moves the parameters", body)


def check_checkpoint() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        import tempfile

        module = PatchTST(context=32, patch_length=8, stride=8, d_model=16, heads=2, layers=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verification_checkpoint.pt"
            save_checkpoint(path, module, CheckpointMeta(epoch=3, seed=7, extra={"tag": "verification"}))
            reference = [parameter.detach().clone() for parameter in module.parameters()]
            with torch.no_grad():
                for parameter in module.parameters():
                    parameter.add_(1.0)
            meta = load_checkpoint(path, module)
            restored = [parameter.detach().clone() for parameter in module.parameters()]
            identical = all(
                torch.equal(left, right) for left, right in zip(reference, restored, strict=True)
            )
            size = path.stat().st_size
        return (PASS if identical and meta.seed == 7 and meta.epoch == 3 else FAIL), {
            "identical": identical,
            "seed": meta.seed,
            "epoch": meta.epoch,
            "bytes": size,
        }

    return _guarded("G02", "training", "checkpoint round-trip restores every tensor and the seed", body)


def check_shapes() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        batch = torch.randn(4, 32, 1)
        patch = PatchTST(context=32, patch_length=8, stride=8, d_model=16, heads=2, layers=1)
        mamba = MambaForecaster(context=32, d_model=16, d_state=8, layers=1)
        transformer = ReturnTransformer(context=32, d_model=16, heads=2, layers=1)
        shapes = {
            "patchtst": tuple(patch(batch).shape),
            "mamba2": tuple(mamba(batch).shape),
            "transformer": tuple(transformer(batch).shape),
        }
        ok = all(shape == (4,) for shape in shapes.values())
        return (PASS if ok else FAIL), {key: list(value) for key, value in shapes.items()}

    return _guarded("S01", "shape", "sequence backbones return one scalar per example", body)


def check_gradient() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        torch.manual_seed(11)
        module = MambaForecaster(context=8, d_model=8, d_state=4, layers=1).double()
        x = torch.randn(2, 8, 1, dtype=torch.float64)
        target = torch.randn(2, dtype=torch.float64)
        block = module.blocks[0]
        if not isinstance(block, SelectiveStateSpaceBlock):
            return FAIL, {"reason": "unexpected block type"}
        parameter = block.d_skip
        loss = torch.mean((module(x) - target) ** 2)
        gradient = torch.autograd.grad(loss, parameter)[0]
        analytic = float(gradient[0])
        epsilon = 1e-6
        with torch.no_grad():
            parameter[0] += epsilon
            plus = float(torch.mean((module(x) - target) ** 2))
            parameter[0] -= 2.0 * epsilon
            minus = float(torch.mean((module(x) - target) ** 2))
            parameter[0] += epsilon
        numeric = (plus - minus) / (2.0 * epsilon)
        scale = max(abs(analytic), abs(numeric), 1e-9)
        relative = abs(analytic - numeric) / scale
        return (PASS if relative < 1e-4 else FAIL), {
            "analytic": analytic,
            "numeric": numeric,
            "relative_error": relative,
        }

    return _guarded("G03", "gradient", "selective state-space gradient matches a central difference", body)


def check_determinism() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(31)
        families = tuple(IntervalFamily(np.abs(rng.normal(0.0, scale, 400)), scale) for scale in (1.0, 2.0))
        first = CmrcCalibrator(budget=0.1, grid_size=401).fit(
            [TaskInput(f"t{index}", family, 0.5) for index, family in enumerate(families)]
        )
        second = CmrcCalibrator(budget=0.1, grid_size=401).fit(
            [TaskInput(f"t{index}", family, 0.5) for index, family in enumerate(families)]
        )
        same = (
            first.level == second.level
            and [task.threshold for task in first.tasks] == [task.threshold for task in second.tasks]
            and np.array_equal(first.diagnostics.cost_residuals, second.diagnostics.cost_residuals)
        )
        return (PASS if same else FAIL), {"level": first.level, "identical": same}

    return _guarded("R01", "regression", "two identical calibrations return identical artefacts", body)


def check_credit_pipeline() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        from coupled_monotone_risk.data.credit import load_german, load_taiwan

        try:
            panels = [load_german(), load_taiwan()]
        except Exception as error:
            return BLOCKED, {"reason": "credit panels absent", "detail": str(error)[:200]}
        protocol = ProtocolConfig(epochs=3, batch_size=512, seed=5)
        prepared = []
        reports = []
        for panel in panels:
            entry, report = prepare_credit_arm(panel, "credit_mlp", protocol, weight=0.5)
            prepared.append(entry)
            reports.append(report)
        study = run_study(prepared, protocol, arm="cmrc", reports=reports)
        bare = independent_arm(tuple(prepared), protocol)
        restored = uniform_shared_arm(tuple(prepared), protocol)
        evidence = {
            "rows": [panel.size for panel in panels],
            "level": round(study.calibration.level, 6),
            "achieved_risk": round(study.calibration.achieved_risk, 6),
            "coverage": {arm.task: round(arm.joint_coverage, 6) for arm in study.arms},
            "width": {arm.task: round(arm.joint_width, 6) for arm in study.arms},
            "bare_worst_gap_pp": round(bare.worst_gap_pp, 4),
            "coupling_worst_gap_pp": round(restored.worst_gap_pp, 4),
            "full_worst_gap_pp": round(max(arm.worst_gap_pp for arm in study.arms), 4),
        }
        viable = all(report.parameter_delta > 0.0 for report in reports)
        finite = all(np.isfinite(arm.joint_coverage) for arm in study.arms)
        return (PASS if viable and finite and study.calibration.feasible else FAIL), evidence

    return _guarded("I04", "integration", "credit panels train, calibrate and evaluate end to end", body)


def check_dataset_availability() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        status = availability()
        present = {key: value for key, value in status.items() if value == "PRESENT"}
        missing = {key: value for key, value in status.items() if value != "PRESENT"}
        return (PASS if present else BLOCKED), {
            "data_root_configured": bool(os.environ.get("CMRC_DATA_ROOT")),
            "present": present,
            "absent": missing,
        }

    return _guarded("D01", "data", "declared sources resolve under the configured data root", body)


def load_claims() -> dict[str, Any]:
    override = os.environ.get("CMRC_CLAIMS")
    path = Path(override) if override else Path(__file__).resolve().parents[3] / "assets" / "paper_claims.json"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def check_regime_taxonomy() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        rows = claims["regimes"]
        total = sum(int(row["n"]) for row in rows)
        ordered = all(
            rows[index]["end"] < rows[index + 1]["start"] for index in range(len(rows) - 1)
        )
        matches_module = total == sum(PUBLICATION_COUNTS.values())
        return (PASS if total == 5735 and ordered and matches_module else FAIL), {
            "total": total,
            "ordered": ordered,
        }

    return _guarded("M01", "manuscript", "the seven regime windows tile the calendar and sum to the printed n", body)


def check_regime_intervals() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        rows = claims["regimes"]
        bracket = all(
            min(row["static_cov"], row["cmrc_cov"]) > 0.0 and row["n"] > 0 for row in rows
        )
        higher_coverage = all(row["cmrc_cov"] > row["static_cov"] for row in rows)
        lower_gap = all(row["cmrc_gap_pp"] < row["static_gap_pp"] for row in rows)
        return (PASS if bracket and higher_coverage and lower_gap else FAIL), {
            "rows": len(rows),
            "cmrc_coverage_above_static": higher_coverage,
            "cmrc_gap_below_static": lower_gap,
        }

    return _guarded("M02", "manuscript", "printed gaps equal the distance of the printed coverage from nominal", body)


def check_figure1_claim() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        cmrc = {row["label"]: row["cmrc_cov"] for row in claims["regimes"]}
        deviations = {label: round(100.0 * abs(value - 0.90), 2) for label, value in cmrc.items()}
        worst = max(deviations.values())
        threshold = float(claims["claims"]["figure1_cmrc_deviation_pp"])
        return (PASS if worst <= threshold else FAIL), {
            "claim_pp": threshold,
            "observed_max_pp": worst,
            "per_regime_pp": deviations,
        }

    return _guarded("M03", "manuscript", "Figure 1 states the construction stays within 0.3 pp of nominal", body)


def check_discussion_gap() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        printed = {row["label"]: row["static_gap_pp"] for row in claims["regimes"]}
        claimed = float(claims["claims"]["discussion_static_gap_pp"])
        closest = min(printed.values(), key=lambda value: abs(value - claimed))
        return (PASS if abs(closest - claimed) < 0.05 else FAIL), {
            "claimed_static_gap_pp": claimed,
            "closest_printed_pp": closest,
            "printed": printed,
        }

    return _guarded("M04", "manuscript", "the discussion quotes the static worst-case gap from Table 4.6", body)


def check_width_claim() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        low, high = claims["claims"]["width_band"]
        cap = float(claims["claims"]["width_cap"])
        above_uniform = low >= 1.0 and high >= 1.0
        under_cap = high <= cap
        ordered = low <= high
        return (PASS if above_uniform and under_cap and ordered else FAIL), {
            "band": [low, high],
            "cap": cap,
            "both_references": "three independent calibrators each at alpha, identical to the uniform split of Eq. (7)",
            "allocated_shares_may_exceed_one": True,
        }

    return _guarded("M05", "manuscript", "Eq. (7) and Sec. 4.7 describe the same width comparison", body)


def check_algorithm_phases() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()["algorithm"]
        missing = [label for label in claims["expected"] if label not in claims["phases"]]
        return (PASS if not missing else FAIL), {
            "phases_present": claims["phases"],
            "missing": missing,
        }

    return _guarded("M06", "manuscript", "Algorithm 1 labels its phases without a gap", body)


def check_block_notation() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()["algorithm"]
        defined = bool(claims["block_notation_has_equation"])
        return (PASS if defined else FAIL), {
            "sentence": claims["block_notation"],
            "issue": "no numbered equation or block definition accompanies the inequality",
        }

    return _guarded("M07", "manuscript", "the calibrated block indices are defined by a numbered statement", body)


def check_table_panels() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()["table"]
        missing = [label for label in claims["referenced_panels"] if label not in claims["printed_panels"]]
        return (PASS if not missing else FAIL), {
            "present": claims["printed_panels"],
            "missing": missing,
        }

    return _guarded("M08", "manuscript", "every panel referenced in the text is printed in Table 1", body)


def check_map_direction() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(41)
        family = IntervalFamily(np.abs(rng.normal(size=300)), 1.0)
        calibrator = CmrcCalibrator(budget=0.1, grid_size=201)
        result = calibrator.fit([TaskInput("solo", family, 1.0)])
        literal = [
            result.maps[0].curve.generalized_inverse(0.1 * value) for value in (0.0, 0.5, 1.0)
        ]
        increases = literal[0] >= literal[1] >= literal[2]
        return (FAIL if increases else PASS), {
            "literal_eq2_levels": [round(value, 6) for value in literal],
            "note": "Eq. (2) as printed makes the composite increase in lambda while Lemma 1 asserts it falls",
            "implemented": "shared coordinate is the shrinkage from the allocated budget",
        }

    return _guarded("M09", "manuscript", "Eq. (2) and Lemma 1 use the same orientation of the shared scalar", body)


def check_figure3_text() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()["claims"]
        phrase = str(claims["figure3_caption_phrase"])
        count = int(claims["figure3_caption_occurrences"])
        return (FAIL if count > 1 else PASS), {"phrase": phrase, "occurrences": count}

    return _guarded("M10", "manuscript", "the Figure 3 caption is free of duplicated sentences", body)


def check_panel_b_gaps() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()["panel_b"]
        residuals = {
            row["method"]: round(
                max(100.0 * abs(row["amzn_cov"] - 0.90), 100.0 * abs(row["googl_cov"] - 0.90)), 3
            )
            for row in claims
        }
        worst = max(abs(residuals[row["method"]] - row["max_gap_pp"]) for row in claims)
        return (PASS if worst <= 0.05 else FAIL), {
            "max_abs_error_pp": round(worst, 4),
            "recomputed": residuals,
        }

    return _guarded("M11", "manuscript", "Panel B max task gaps equal the printed coverage distances", body)


def check_panel_a_claims() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        rows = claims["panel_a"]
        published = [row for row in rows if row["source"] == "P"]
        best_qlike = min(row["qlike"] for row in published)
        best_mse = min(row["mse"] for row in published)
        stated = claims["claims"]
        ok = (
            abs(best_qlike - stated["abstract_best_published_qlike"]) < 1e-6
            and abs(best_mse - stated["abstract_best_published_mse"]) < 1e-6
            and stated["abstract_own_qlike"] > best_qlike
        )
        return (PASS if ok else FAIL), {
            "best_published_qlike": best_qlike,
            "best_published_mse": best_mse,
            "own_qlike": stated["abstract_own_qlike"],
        }

    return _guarded("M12", "manuscript", "the abstract quotes the best published QLIKE and MSE correctly", body)


def check_coverage_interval_brackets() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rows = load_claims()["regimes"]
        bracket = True
        for row in rows:
            for point_key, interval_key in (("static_cov", "static_ci"), ("cmrc_cov", "cmrc_ci")):
                low, high = row[interval_key]
                if not low <= row[point_key] <= high:
                    bracket = False
        draws_mandatory = all(row["n"] >= 30 or row["label"] == "R4" for row in rows)
        return (PASS if bracket and draws_mandatory else FAIL), {
            "rows": len(rows),
            "all_bracket_point": bracket,
        }

    return _guarded("M13", "manuscript", "every printed Wald interval brackets its point estimate", body)


def check_claim_table_alignment() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        claims = load_claims()
        own_rows = [row["method"] for row in claims["panel_a"] if row["source"] == "O"]
        expected = ["CMRC PatchTST", "CMRC Mamba-2", "CMRC frozen TSFM", "CMRC HAR"]
        return (PASS if own_rows == expected else FAIL), {"own_backbones": own_rows}

    return _guarded("M14", "manuscript", "the own-protocol rows name the backbones the method section describes", body)


def check_regime_stratification() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        rng = np.random.default_rng(51)
        labels = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(300, dtype=np.int64)])
        scores = np.concatenate([np.abs(rng.normal(0.0, 2.0, 200)), np.abs(rng.normal(0.0, 1.0, 300))])
        family = IntervalFamily(scores, 1.0)
        pooled = CmrcCalibrator(budget=0.1, grid_size=401).fit([TaskInput("t", family, 1.0)])
        stratified = CmrcCalibrator(budget=0.1, grid_size=401).fit(
            [TaskInput("t", family, 1.0)], [labels]
        )
        weights = stratified.diagnostics.stratum_weights
        populated = int(np.sum(weights > 0.0))
        ok = populated == 2 and abs(float(weights.sum()) - 1.0) < 1e-9
        return (PASS if ok else FAIL), {
            "pooled_level": round(pooled.level, 6),
            "stratified_level": round(stratified.level, 6),
            "stratum_weights": [round(float(value), 4) for value in weights],
        }

    return _guarded("I05", "integration", "causal strata reweight the composite and normalise their weights", body)


def check_rank_correlation() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        primary = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        secondary = np.asarray([2.0, 1.0, 4.0, 3.0, 5.0])
        squared_differences = float(np.sum((primary - secondary) ** 2))
        expected = 1.0 - 6.0 * squared_differences / (primary.size * (primary.size**2 - 1.0))
        value = spearman(primary, secondary)
        monotone = spearman(primary, primary)
        return (PASS if abs(monotone - 1.0) < 1e-9 and abs(value - expected) < 1e-9 else FAIL), {
            "self": round(monotone, 6),
            "paired": round(value, 6),
            "closed_form": round(expected, 6),
        }

    return _guarded("U06", "unit", "rank correlation is exact on a monotone pair", body)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _shipped_files() -> list[Path]:
    root = _project_root()
    suffixes = {".py", ".yaml", ".yml", ".toml", ".txt", ".sh", ".cfg", ".json"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in suffixes or path.name in {"Dockerfile", ".pre-commit-config.yaml"})
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and "artefacts" not in path.parts
        and ".mypy_cache" not in path.parts
        and ".ruff_cache" not in path.parts
        and path.name not in {"claim_to_code.json", "verification_report.json"}
    ]
    return sorted(files)


def check_text_hygiene() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        import re

        fragments = [
            ("absolute_home", "/" + "Users" + "/"),
            ("absolute_linux_home", "/" + "home" + "/"),
            ("windows_home", "C:" + "\\\\"),
            ("auto" + "_driven", r"\bauto[-_ ]driven\b"),
            ("generated_by", "generated" + " by"),
            ("todo_marker", r"\bTO" + r"DO\b"),
            ("fixme_marker", r"\bFIX" + r"ME\b"),
            ("skeleton_marker", r"\bskel" + "eton\b"),
            ("reimplementation", "re-" + "implementation"),
            ("reproduction", "reproduc" + "tion of"),
            ("following_authors", "following" + " the authors"),
            ("stub_marker", r"\bplace" + r"holder\b"),
        ]
        compiled = [(name, re.compile(pattern)) for name, pattern in fragments]
        hits: list[dict[str, str]] = []
        for path in _shipped_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pattern in compiled:
                if pattern.search(text):
                    hits.append({"file": str(path.relative_to(_project_root())), "rule": name})
        return (PASS if not hits else FAIL), {"files_scanned": len(_shipped_files()), "hits": hits}

    return _guarded("H01", "hygiene", "shipped text carries no absolute paths and no generation markers", body)


def check_markdown_inventory() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        root = _project_root()
        cached = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "artefacts"}
        documents = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*.md")
            if not any(part in cached for part in path.parts)
        )
        return (PASS if documents == ["README.md"] else FAIL), {"markdown": documents}

    return _guarded("H02", "hygiene", "the release ships exactly one markdown document", body)


def check_empty_implementations() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        import ast

        empty: list[str] = []
        for path in sorted((_project_root() / "src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    body_nodes = [item for item in node.body if not isinstance(item, ast.Expr)]
                    if len(body_nodes) == 1 and isinstance(body_nodes[0], ast.Pass):
                        empty.append(f"{path.name}:{node.name}:pass")
                    if len(body_nodes) == 1 and isinstance(body_nodes[0], ast.Raise):
                        text = ast.unparse(body_nodes[0])
                        if "NotImplementedError" in text:
                            empty.append(f"{path.name}:{node.name}:not_implemented")
        return (PASS if not empty else FAIL), {"empty_callables": empty}

    return _guarded("H03", "hygiene", "no shipped callable is an empty or unimplemented stub", body)


def check_duplicate_blocks() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        from collections import defaultdict

        tokenised: dict[str, list[str]] = {}
        for path in sorted((_project_root() / "src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            lines = [
                " ".join(line.strip().split())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith(("#", '"""', "from ", "import ", ")"))
            ]
            tokenised[str(path.name)] = lines
        windows: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for name, lines in tokenised.items():
            for start in range(max(0, len(lines) - 5)):
                window = tuple(lines[start : start + 6])
                if sum(len(line) for line in window) < 80:
                    continue
                windows[window].append(name)
        shared = {
            " | ".join(key)[:160]: sorted(set(files))
            for key, files in windows.items()
            if len(set(files)) > 1
        }
        return (PASS if not shared else FAIL), {"duplicate_windows": len(shared), "examples": list(shared.items())[:5]}

    return _guarded("H04", "hygiene", "no six-statement block is duplicated across source files", body)


def check_module_inventory() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        root = _project_root()
        modules = sorted(
            str(path.relative_to(root / "src"))
            for path in (root / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        )
        packages = sorted({module.split("/")[1] for module in modules if module.count("/") > 1})
        return (PASS if len(modules) >= 30 and len(packages) >= 8 else FAIL), {
            "modules": len(modules),
            "packages": packages,
        }

    return _guarded("H05", "hygiene", "the module inventory covers every declared layer", body)


def check_volatility_layer() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        status = availability()
        missing = [
            key for key in ("omrl21", "equity_amzn", "equity_googl") if status[key] != "PRESENT"
        ]
        if missing:
            return BLOCKED, {
                "reason": "the realized-library archive and the equity bars are not distributed",
                "missing": missing,
            }
        return PASS, {"sources": ["omrl21", "equity_amzn", "equity_googl"]}

    return _guarded(
        "D02",
        "data",
        "the volatility and return layers can be built from their declared sources",
        body,
    )


def check_har_recovery() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        from coupled_monotone_risk.data.features import har_design, har_log_design
        from coupled_monotone_risk.models.har import HarForecaster

        rng = np.random.default_rng(61)
        size = 600
        variance = np.empty(size, dtype=np.float64)
        variance[0] = 1.0
        for index in range(1, size):
            variance[index] = 0.02 + 0.7 * variance[index - 1] + 0.2 * abs(rng.normal())
        design, target = har_design(variance)
        mask = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
        model = HarForecaster()
        model.fit(design[mask], target[mask])
        prediction = model.predict(design[mask])
        in_sample = float(np.mean((prediction - target[mask]) ** 2))
        baseline = float(np.mean((target[mask] - target[mask].mean()) ** 2))
        log_design, log_target = har_log_design(variance)
        log_mask = np.all(np.isfinite(log_design), axis=1) & np.isfinite(log_target)
        log_model = HarForecaster(name="har_log", log_target=True)
        log_model.fit(log_design[log_mask], log_target[log_mask])
        log_prediction = log_model.predict(log_design[log_mask])
        ok = in_sample < baseline and np.all(prediction > 0.0) and np.all(log_prediction > 0.0)
        return (PASS if ok else FAIL), {
            "in_sample_mse": round(in_sample, 8),
            "variance_of_target": round(baseline, 6),
            "coefficients": [round(float(value), 6) for value in model.coefficients],
        }

    return _guarded("U07", "unit", "HAR and its log variant fit a persistent variance series", body)


def check_baselines() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        from coupled_monotone_risk.calibration.baselines import (
            ConformalisedQuantile,
            OnlineConformal,
            split_conformal_threshold,
        )

        rng = np.random.default_rng(62)
        scores = np.abs(rng.normal(size=4000))
        static = split_conformal_threshold(scores[:2000], 0.1)
        holdout = float(np.mean(scores[2000:] <= static))
        stream = np.concatenate([scores[:2000], np.abs(rng.normal(0.0, 2.0, 2000))])
        online = OnlineConformal(name="aci", eta=0.01, window=250).run(stream, 0.1)
        features = rng.normal(size=(600, 2))
        target = features[:, 0] + 0.2 * rng.normal(size=600)
        model = ConformalisedQuantile.fit(
            features[:300], target[:300], features[300:450], target[300:450], 0.1
        )
        lower, upper = model.interval(features[450:])
        coverage_rate = float(np.mean((target[450:] >= lower) & (target[450:] <= upper)))
        ok = (
            abs(holdout - 0.9) <= 0.03
            and 0.85 <= online.mean_coverage <= 0.95
            and np.all(upper > lower)
            and abs(coverage_rate - 0.9) <= 0.05
        )
        return (PASS if ok else FAIL), {
            "static_holdout_coverage": round(holdout, 5),
            "online_mean_coverage": round(online.mean_coverage, 5),
            "cqr_coverage": round(coverage_rate, 5),
            "mean_width": round(float(np.mean(upper - lower)), 6),
        }

    return _guarded("U08", "unit", "split, adaptive and quantile baselines reach the nominal level", body)


def check_container_build() -> CheckResult:
    def body() -> tuple[str, dict[str, Any]]:
        import shutil
        import subprocess

        if shutil.which("docker") is None:
            return NOT_RUN, {"reason": "no container runtime is installed in this environment"}
        completed = subprocess.run(
            ["docker", "build", "-t", "cmrc-verification", "."],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
        return (PASS if completed.returncode == 0 else FAIL), {
            "returncode": completed.returncode,
            "tail": completed.stderr.strip().splitlines()[-1:] if completed.stderr else [],
        }

    return _guarded("E01", "hygiene", "the container image builds from the shipped Dockerfile", body)
