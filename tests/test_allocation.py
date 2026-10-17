import numpy as np
import pytest

from coupled_monotone_risk.coupling.allocation import (
    equalised_marginal_allocation,
    marginal_cost_spread,
    uniform_allocation,
)
from coupled_monotone_risk.coupling.composite import (
    CompositeRisk,
    StratifiedView,
    find_crossing,
)
from coupled_monotone_risk.coupling.maps import build_maps
from coupled_monotone_risk.risks.cost import CostCurve
from coupled_monotone_risk.risks.curve import RiskCurve
from coupled_monotone_risk.risks.family import IntervalFamily


def _costs(pair: tuple[IntervalFamily, IntervalFamily]) -> tuple[CostCurve, CostCurve]:
    first, second = pair
    return (
        CostCurve.from_curve(RiskCurve.from_family(first, 401)),
        CostCurve.from_curve(RiskCurve.from_family(second, 401)),
    )


def test_allocation_satisfies_the_budget_identity(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    curves = _costs(interval_pair)
    weights = np.asarray([0.6, 0.4])
    result = equalised_marginal_allocation(curves, weights, 0.1)
    assert float(np.dot(weights, result.shares)) == pytest.approx(1.0, abs=1e-9)
    assert float(np.dot(weights, result.budgets)) == pytest.approx(0.1, abs=1e-9)


def test_allocation_beats_the_uniform_split(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    curves = _costs(interval_pair)
    result = equalised_marginal_allocation(curves, np.asarray([0.5, 0.5]), 0.1)
    assert result.allocated_objective <= result.uniform_objective + 1e-9


def test_marginal_spread_is_one_for_proportional_curves(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    family, _ = interval_pair
    curve = CostCurve.from_curve(RiskCurve.from_family(family, 401))
    assert marginal_cost_spread((curve, curve), 0.1) == pytest.approx(1.0)


def test_uniform_allocation_is_declared_as_such() -> None:
    result = uniform_allocation(3, 0.1)
    assert np.allclose(result.shares, 1.0)
    assert np.allclose(result.budgets, 0.1)
    assert result.converged


def test_allocation_rejects_negative_budget(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    curves = _costs(interval_pair)
    with pytest.raises(ValueError):
        equalised_marginal_allocation(curves, np.asarray([0.5, 0.5]), -0.1)


def test_maps_are_increasing_and_injective(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    curves = (
        RiskCurve.from_family(first, 301),
        RiskCurve.from_family(second, 301),
    )
    maps = build_maps(("a", "b"), curves, np.asarray([0.5, 0.5]), 0.1)
    positive, strict = maps[0].monotone_profile()
    assert positive == 1.0
    assert strict > 0.9
    assert maps[0].injective_share() > 0.99
    assert maps[0].saturation_share() < 0.2


def test_composite_risk_falls_along_the_axis(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    curves = (
        RiskCurve.from_family(first, 301),
        RiskCurve.from_family(second, 301),
    )
    maps = build_maps(("a", "b"), curves, np.asarray([0.5, 0.5]), 0.1)
    composite = CompositeRisk(maps=maps, weights=np.asarray([0.5, 0.5]), budget=0.1)
    levels = np.linspace(0.0, 1.0, 21)
    values = [composite.risk(float(level)) for level in levels]
    assert all(values[index] >= values[index + 1] - 1e-9 for index in range(len(values) - 1))


def test_crossing_returns_the_first_feasible_level() -> None:
    result = find_crossing(lambda level: 0.4 * (1.0 - level), 0.1, 500, grid_size=201)
    assert result.feasible
    assert result.shifted_risk <= 0.1 + 1e-9
    assert 0.7 <= result.level <= 0.8


def test_crossing_reports_infeasibility() -> None:
    result = find_crossing(lambda level: 0.5 + 0.0 * level, 0.1, 500, grid_size=101)
    assert not result.feasible
    assert result.achieved_risk == pytest.approx(0.5)


def test_stratified_view_normalises_weights(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    curves = (
        RiskCurve.from_family(first, 201),
        RiskCurve.from_family(second, 201),
    )
    maps = build_maps(("a", "b"), curves, np.asarray([0.5, 0.5]), 0.1)
    composite = CompositeRisk(maps=maps, weights=np.asarray([0.5, 0.5]), budget=0.1)
    labels = (
        np.asarray([0] * 300 + [1] * 300, dtype=np.int64),
        np.asarray([0] * 200 + [1] * 250, dtype=np.int64),
    )
    view = StratifiedView.from_labels(composite, labels, 2)
    assert float(view.weights.sum()) == pytest.approx(1.0)
    assert view.risk(0.0) == pytest.approx(composite.risk(0.0), abs=0.05)
