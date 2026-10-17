import numpy as np
import pytest

from coupled_monotone_risk.risks.cost import CostCurve, uniform_width
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.risks.family import (
    IntervalFamily,
    SetFamily,
    family_from_predictions,
)
from coupled_monotone_risk.risks.isotonic import fit_decreasing, fit_increasing


def test_interval_family_risk_decreases(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    risks = [family.risk(level) for level in np.linspace(0.0, 1.0, 11)]
    assert all(risks[index] >= risks[index + 1] - 1e-12 for index in range(len(risks) - 1))


def test_interval_width_is_twice_the_quantile(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    assert family.width(0.5) == pytest.approx(2.0 * family.threshold(0.5))


def test_family_rejects_negative_residuals() -> None:
    with pytest.raises(ValueError):
        IntervalFamily(np.asarray([1.0, -1.0]), 1.0)


def test_set_family_width_is_at_least_one(mixed_pair: tuple[IntervalFamily, SetFamily]) -> None:
    _, family = mixed_pair
    assert family.width(1.0) >= 1.0
    assert family.risk(1.0) == 0.0


def test_set_prediction_sets_are_boolean(mixed_pair: tuple[IntervalFamily, SetFamily]) -> None:
    _, family = mixed_pair
    sets = family.predicted_sets(0.9)
    assert sets.dtype == np.bool_
    assert sets.shape[0] == family.size


def test_family_factory_rejects_unknown_kind(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        family_from_predictions("interval", rng.normal(size=10), rng.normal(size=10), None)
    with pytest.raises(ValueError):
        family_from_predictions("tree", rng.normal(size=10), rng.normal(size=10))


def test_curve_inverse_is_continuous(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    curve = RiskCurve.from_family(family, 501)
    left = curve.generalized_inverse(0.1)
    right = curve.generalized_inverse(0.1001)
    assert abs(left - right) < 0.05


def test_curve_lipschitz_and_slope_are_positive(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    curve = RiskCurve.from_family(family, 401)
    assert curve.lipschitz() > 0.0
    assert curve.slope() > 0.0
    assert curve.monotone_share() == 1.0


def test_curve_masked_risk_uses_the_mask(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    curve = RiskCurve.from_family(family, 301)
    mask = np.zeros(family.size, dtype=bool)
    mask[:100] = True
    assert curve.masked_risk(mask, 0.5) == pytest.approx(family.risk(0.5, mask))


def test_cost_curve_is_decreasing(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    cost = CostCurve.from_curve(RiskCurve.from_family(family, 401))
    assert np.all(np.diff(cost.fitted) <= 1e-12)
    assert cost.width_at(0.05) > cost.width_at(0.3)


def test_cost_marginal_is_positive(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    cost = CostCurve.from_curve(RiskCurve.from_family(family, 401))
    assert cost.marginal(0.1) >= 0.0
    assert uniform_width(cost, 0.1) == pytest.approx(cost.width_at(0.1))


def test_isotonic_residual_is_small_for_a_monotone_sequence() -> None:
    x = np.linspace(0.0, 1.0, 20)
    fit = fit_decreasing(x, 1.0 - x)
    assert fit.residual < 1e-12
    assert fit_increasing(x, x).residual < 1e-12


def test_curve_rejects_unsorted_levels(interval_pair: tuple[IntervalFamily, IntervalFamily]) -> None:
    family, _ = interval_pair
    curve = RiskCurve.from_family(family, 51)
    with pytest.raises(ValueError):
        RiskCurve(levels=curve.levels[::-1], risk=curve.risk, width=curve.width, family=family)
