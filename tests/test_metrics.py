import numpy as np
import pytest

from coupled_monotone_risk.metrics.backtest import (
    binomial_interval,
    christoffersen_conditional,
    christoffersen_independence,
    failure_sequence,
    kupiec_pof,
)
from coupled_monotone_risk.metrics.classification import (
    auc_roc,
    average_precision,
    brier_score,
    expected_calibration_error,
    ks_statistic,
    prediction_set_size,
)
from coupled_monotone_risk.metrics.interval import (
    coverage,
    crps_from_pinball,
    mean_width,
    pinball,
    wald_interval,
    winkler_score,
)
from coupled_monotone_risk.metrics.point import (
    directional_accuracy,
    mean_absolute_deviation,
    mean_squared_error,
    qlike,
)
from coupled_monotone_risk.metrics.ranking import diebold_mariano, model_confidence_set


def test_qlike_is_zero_for_a_perfect_forecast() -> None:
    actual = np.asarray([0.1, 0.2, 0.3])
    assert qlike(actual, actual) == pytest.approx(0.0)


def test_qlike_punishes_under_prediction_more_than_over_prediction() -> None:
    actual = np.asarray([0.2, 0.2])
    under = qlike(actual, np.asarray([0.1, 0.1]))
    over = qlike(actual, np.asarray([0.4, 0.4]))
    assert under > over


def test_mse_and_mad_are_zero_for_a_perfect_forecast() -> None:
    actual = np.asarray([0.1, 0.4])
    assert mean_squared_error(actual, actual) == pytest.approx(0.0)
    assert mean_absolute_deviation(actual, actual) == pytest.approx(0.0)


def test_directional_accuracy_matches_signs() -> None:
    previous = np.asarray([1.0, 1.0, 1.0])
    actual = np.asarray([2.0, 0.5, 1.0])
    forecast = np.asarray([1.5, 0.8, 2.0])
    assert directional_accuracy(actual, forecast, previous) == pytest.approx(1.0)
    reversed_forecast = np.asarray([1.5, 1.2, 2.0])
    assert directional_accuracy(actual, reversed_forecast, previous) == pytest.approx(0.5)


def test_coverage_counts_inclusive_ends() -> None:
    lower = np.asarray([0.0, 0.0])
    upper = np.asarray([1.0, 1.0])
    actual = np.asarray([0.0, 1.0])
    assert coverage(lower, upper, actual) == pytest.approx(1.0)
    assert mean_width(lower, upper) == pytest.approx(1.0)


def test_winkler_is_zero_for_a_degenerate_correct_interval() -> None:
    actual = np.asarray([0.5, 0.5])
    assert winkler_score(actual, actual, actual, 0.1) == pytest.approx(0.0)


def test_pinball_and_crps_are_zero_for_a_perfect_quantile() -> None:
    actual = np.asarray([0.5, 0.7])
    quantiles = np.column_stack([actual, actual])
    assert pinball(actual, actual, 0.5) == pytest.approx(0.0)
    assert crps_from_pinball(actual, quantiles, np.asarray([0.1, 0.9])) == pytest.approx(0.0)


def test_wald_interval_shrinks_with_the_sample() -> None:
    narrow = wald_interval(np.asarray([1, 0] * 500))
    wide = wald_interval(np.asarray([1, 0] * 5))
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_kupiec_accepts_the_nominal_rate() -> None:
    outcome = kupiec_pof(10, 100, 0.10)
    assert outcome.p_value > 0.9


def test_kupiec_rejects_a_double_rate() -> None:
    outcome = kupiec_pof(20, 100, 0.10)
    assert outcome.p_value < 0.01


def test_christoffersen_detects_clustering() -> None:
    rng = np.random.default_rng(0)
    independent = rng.integers(0, 2, size=200).astype(np.int64)
    clustered = np.asarray([0] * 60 + [1] * 60 + [0] * 60, dtype=np.int64)
    assert christoffersen_independence(clustered).statistic > christoffersen_independence(independent).statistic


def test_christoffersen_conditional_combines_two_tests() -> None:
    hits = np.asarray([0, 0, 1, 0, 0, 1] * 20, dtype=np.int64)
    outcome = christoffersen_conditional(hits)
    assert outcome.degrees == 2
    assert 0.0 <= outcome.p_value <= 1.0


def test_failure_sequence_flags_outside_values() -> None:
    flags = failure_sequence(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0]), np.asarray([0.5, 1.5]))
    assert flags.tolist() == [0, 1]


def test_binomial_interval_is_ordered() -> None:
    low, high = binomial_interval(np.asarray([1, 1, 0, 1]))
    assert low <= high


def test_auc_and_friends_on_a_separable_problem() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auc_roc(labels, scores) == pytest.approx(1.0)
    assert average_precision(labels, scores) == pytest.approx(1.0)
    assert ks_statistic(labels, scores) == pytest.approx(1.0)


def test_brier_and_ece_on_calibrated_probabilities() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert brier_score(labels, probabilities) == pytest.approx(0.0)
    assert expected_calibration_error(labels, probabilities, bins=4) == pytest.approx(0.0)


def test_prediction_set_size_counts_members() -> None:
    mask = np.asarray([[True, False], [True, True]])
    assert prediction_set_size(mask) == pytest.approx(1.5)


def test_diebold_mariano_detects_a_clear_difference() -> None:
    rng = np.random.default_rng(2)
    better = rng.normal(0.0, 0.1, 400)
    worse = better + 0.2
    outcome = diebold_mariano(worse, better, horizon=1)
    assert outcome.mean_difference == pytest.approx(0.2, abs=1e-6)
    assert outcome.p_value < 0.01


def test_model_confidence_set_keeps_the_best_model() -> None:
    rng = np.random.default_rng(3)
    losses = np.column_stack(
        [rng.normal(0.0, 0.1, 300), rng.normal(1.0, 0.1, 300), rng.normal(2.0, 0.1, 300)]
    )
    result = model_confidence_set(losses, alpha=0.1, bootstrap=100, seed=5)
    assert 0 in result.survivors
    assert 2 in result.eliminated
