import math

import numpy as np
import pytest

from coupled_monotone_risk.bounds.finite_sample import (
    BoundInputs,
    estimation_slack,
    sampling_slack,
    theorem1_bound,
)
from coupled_monotone_risk.bounds.inflation import (
    VOID_THRESHOLD,
    inflation_bounds,
    measured_exponent,
    residual_rank_correlation,
    screen_alignment,
)
from coupled_monotone_risk.calibration.adaptive import (
    AdaptiveStratumWeights,
    drift_term,
    time_averaged_risk,
)
from coupled_monotone_risk.calibration.shift_band import ShiftBand


def _inputs(sample_size: int = 800, delta: float = 0.05) -> BoundInputs:
    return BoundInputs(
        sample_size=sample_size,
        task_count=3,
        grid_size=1001,
        budget=0.1,
        delta=delta,
        bound=1.0,
        weights=np.asarray([0.4, 0.3, 0.3]),
        lipschitz=np.asarray([1.0, 1.0, 1.0]),
        slope=np.asarray([0.8, 0.9, 0.7]),
        share_errors=np.asarray([0.01, 0.02, 0.015]),
        weight_error=0.03,
    )


def test_sampling_slack_matches_the_closed_form() -> None:
    inputs = _inputs()
    expected = math.sqrt((math.log(1001) + math.log(20.0) + math.log(3)) / (2 * 800))
    assert sampling_slack(inputs) == pytest.approx(expected)


def test_slack_falls_with_sample_size() -> None:
    assert sampling_slack(_inputs(400)) > sampling_slack(_inputs(1600))


def test_slack_grows_with_the_grid() -> None:
    coarse = _inputs()
    fine = BoundInputs(
        sample_size=coarse.sample_size,
        task_count=coarse.task_count,
        grid_size=10001,
        budget=coarse.budget,
        delta=coarse.delta,
        bound=coarse.bound,
        weights=coarse.weights,
        lipschitz=coarse.lipschitz,
        slope=coarse.slope,
        share_errors=coarse.share_errors,
        weight_error=coarse.weight_error,
    )
    assert sampling_slack(fine) > sampling_slack(coarse)


def test_estimation_slack_is_additive() -> None:
    inputs = _inputs()
    report = theorem1_bound(inputs)
    assert report.estimation_slack == pytest.approx(estimation_slack(inputs))
    assert report.total_slack == pytest.approx(report.sampling_slack + report.estimation_slack)
    assert report.certificate > inputs.budget


def test_inflation_bounds_bracket_the_exponent() -> None:
    bounds = inflation_bounds(np.arange(1, 6))
    assert bounds.exponent_shared == 0.0
    assert bounds.exponent_simultaneous == 1.0
    assert bounds.shared[0] == pytest.approx(bounds.simultaneous[0])
    assert bounds.shared[-1] < bounds.simultaneous[-1]


def test_measured_exponent_recovers_one() -> None:
    counts = np.arange(1, 6, dtype=np.float64)
    budgets = 0.01 * counts
    assert measured_exponent(counts, budgets) == pytest.approx(1.0, abs=1e-6)


def test_alignment_screen_voids_below_the_threshold() -> None:
    assert screen_alignment(0.29).void
    assert not screen_alignment(0.41).void
    assert screen_alignment(0.41).verdict == "ADMISSIBLE"
    assert math.isclose(VOID_THRESHOLD, 0.30)


def test_residual_rank_correlation_is_bounded() -> None:
    rng = np.random.default_rng(9)
    timing = np.arange(300, dtype=np.float64)
    residuals = np.column_stack([timing + rng.normal(0, 1, 300), 2 * timing + rng.normal(0, 1, 300)])
    value = residual_rank_correlation(residuals, timing)
    assert -1.0 <= value <= 1.0


def test_adaptive_weights_stay_on_the_simplex() -> None:
    weights = AdaptiveStratumWeights(np.asarray([0.2, 0.3, 0.5]), 0.5)
    updated = weights.update(np.asarray([0.1, 0.9, 0.2]))
    assert float(updated.sum()) == pytest.approx(1.0)
    assert updated[1] < 0.3


def test_adaptive_drift_is_bounded() -> None:
    weights = AdaptiveStratumWeights(np.asarray([0.5, 0.5]), 0.1)
    for _ in range(5):
        weights.update(np.asarray([0.0, 1.0]))
    assert drift_term(weights.history) > 0.0


def test_long_run_bound_reports_excess() -> None:
    weights = AdaptiveStratumWeights(np.asarray([0.5, 0.5]), 0.2)
    bound = weights.long_run(average_risk=0.12, budget=0.1, drift=0.01)
    assert bound.excess == pytest.approx(0.02)
    assert bound.allowance == pytest.approx(0.05)


def test_time_averaged_risk() -> None:
    assert time_averaged_risk(np.asarray([0.1, 0.2, 0.3])) == pytest.approx(0.2)


def test_shift_band_propagation() -> None:
    band = ShiftBand(band=0.02, bound=1.0)
    assert band.propagation_error == pytest.approx(0.02)
    assert band.certificate_level(0.1) == pytest.approx(0.12)
    assert band.coverage_floor(0.9) == pytest.approx(1.0 - 0.12)
