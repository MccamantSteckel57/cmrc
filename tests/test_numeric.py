import math

import numpy as np
import pytest

from coupled_monotone_risk.utils.numeric import (
    as_float_array,
    bisect_monotone,
    check_weights,
    conformal_quantile,
    empirical_quantile,
    invert_decreasing,
    pava_decreasing,
    pava_increasing,
    piecewise_linear_slope,
    spearman,
    stable_softmax,
    weighted_mean,
)


def test_quantiles_recover_order_statistics() -> None:
    values = np.arange(1.0, 11.0)
    assert empirical_quantile(values, 0.0) == 1.0
    assert empirical_quantile(values, 1.0) == 10.0
    assert empirical_quantile(values, 0.5) == 5.0
    rank = math.ceil(11 * 0.9)
    assert conformal_quantile(values, 0.9) == values[rank - 1]


def test_conformal_quantile_is_infinite_at_full_level() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    assert conformal_quantile(values, 1.0) == float("inf")


def test_pava_preserves_the_mean() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 4.0])
    fitted = pava_decreasing(values)
    assert np.all(np.diff(fitted) <= 1e-12)
    assert fitted.mean() == pytest.approx(values.mean())


def test_pava_increasing_is_the_mirror() -> None:
    values = np.asarray([4.0, 2.0, 3.0])
    fitted = pava_increasing(values)
    assert np.all(np.diff(fitted) >= -1e-12)


def test_bisection_finds_a_monotone_root() -> None:
    root, iterations, converged = bisect_monotone(lambda x: 1.0 - x, 0.25, 0.0, 1.0, 1e-9, 80)
    assert converged
    assert root == pytest.approx(0.75, abs=1e-6)
    assert iterations < 80


def test_bisection_rejects_bad_brackets() -> None:
    with pytest.raises(ValueError):
        bisect_monotone(lambda x: x, 0.5, 1.0, 0.0, 1e-6, 10)


def test_inversion_of_a_step_function_stays_inside_the_support() -> None:
    x = np.linspace(0.0, 1.0, 11)
    y = np.asarray([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0])
    assert invert_decreasing(x, y, 1.5) == 0.0
    assert invert_decreasing(x, y, -1.0) == 1.0
    assert 0.2 <= invert_decreasing(x, y, 0.4) <= 0.5


def test_weight_validation() -> None:
    assert check_weights([0.25, 0.75]).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        check_weights([0.5, 0.4])
    with pytest.raises(ValueError):
        check_weights([-0.5, 1.5])


def test_array_validation_rejects_shapes_and_nan() -> None:
    with pytest.raises(ValueError):
        as_float_array(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        as_float_array([1.0, float("nan")])


def test_softmax_normalises() -> None:
    probabilities = stable_softmax(np.asarray([1.0, 2.0, 3.0]))
    assert probabilities.sum() == pytest.approx(1.0)


def test_rank_correlation_bounds() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert spearman(values, values) == pytest.approx(1.0)
    assert spearman(values, values[::-1]) == pytest.approx(-1.0)


def test_piecewise_slope_and_weighted_mean() -> None:
    slopes = piecewise_linear_slope(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 2.0, 4.0]))
    assert np.allclose(slopes, 2.0)
    assert weighted_mean(np.asarray([1.0, 3.0]), np.asarray([0.5, 0.5])) == pytest.approx(2.0)
