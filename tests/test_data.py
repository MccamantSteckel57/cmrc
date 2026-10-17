from datetime import date

import numpy as np
import pytest

from coupled_monotone_risk.data.credit import Standardiser, build_aggregate
from coupled_monotone_risk.data.features import (
    absolute_return_design,
    finite_rows,
    har_design,
    trailing_mean,
    windowed_batches,
)
from coupled_monotone_risk.data.manifest import DATASETS, availability, verified_links
from coupled_monotone_risk.data.regimes import (
    PUBLICATION_COUNTS,
    REGIME_WINDOWS,
    coverage_report,
    label_for,
    label_series,
    taxonomy_sums_to_published_total,
    window_of,
)
from coupled_monotone_risk.data.splits import plan_blocks, rolling_windows, slice_series
from coupled_monotone_risk.data.strata import (
    assign_strata,
    expanding_drawdown,
    stratum_slices,
    stratum_weights,
)


def test_regime_lookup_covers_every_window() -> None:
    assert label_for(date(2008, 5, 1)) == "R2"
    assert label_for(date(2020, 3, 1)) == "R4"
    assert label_for(date(2024, 1, 2)) == "R7"
    with pytest.raises(ValueError):
        label_for(date(1990, 1, 1))


def test_regime_counts_match_the_published_total() -> None:
    assert taxonomy_sums_to_published_total()
    assert sum(PUBLICATION_COUNTS.values()) == 5735


def test_stress_labels_are_the_three_drawdown_windows() -> None:
    stressed = [window.label for window in REGIME_WINDOWS if window.stressed]
    assert stressed == ["R2", "R4", "R6"]
    assert window_of("R4").drawdown == pytest.approx(-0.339)


def test_label_series_and_coverage_report() -> None:
    days = [date(2008, 5, 1), date(2008, 5, 2), date(2020, 3, 2)]
    series = label_series(days)
    assert list(series) == ["R2", "R2", "R4"]
    report = coverage_report(days)
    assert report["R2"] == 2


def test_block_plan_orders_the_four_segments() -> None:
    layout = plan_blocks(1000, 0.6, 0.2, 22)
    assert layout.train()[-1] < layout.embargo_block()[0]
    assert layout.embargo_block()[-1] < layout.calibration()[0]
    assert layout.calibration()[-1] < layout.test()[0]
    assert layout.test().size == 200


def test_block_plan_rejects_impossible_fractions() -> None:
    with pytest.raises(ValueError):
        plan_blocks(1000, 0.9, 0.2, 22)
    with pytest.raises(ValueError):
        plan_blocks(100, 0.5, 0.2, 80)


def test_rolling_windows_respect_the_embargo() -> None:
    windows = rolling_windows(600, window=250, embargo=22, step=50)
    assert windows
    for window in windows:
        assert window.calibration_stop + 22 >= window.test_start
        assert window.calibration_size == 228


def test_slice_series_returns_selected_values() -> None:
    values = np.arange(10, dtype=np.float64)
    assert slice_series(values, np.asarray([1, 3])).tolist() == [1.0, 3.0]


def test_trailing_mean_uses_past_values_only() -> None:
    values = np.arange(1.0, 11.0)
    means = trailing_mean(values, 3)
    assert np.isnan(means[1])
    assert means[2] == pytest.approx(2.0)
    assert means[9] == pytest.approx(9.0)


def test_har_design_columns_have_the_expected_lags() -> None:
    variance = np.arange(1.0, 60.0)
    design, target = har_design(variance, horizon=2)
    assert design.shape == (59, 4)
    assert design[30, 1] == pytest.approx(variance[30])
    assert design[30, 2] == pytest.approx(variance[26:31].mean())
    assert target[30] == pytest.approx(variance[32])


def test_finite_rows_drop_warmup() -> None:
    design, target = har_design(np.arange(1.0, 40.0))
    rows = finite_rows(design, target)
    assert rows[0] >= 21


def test_absolute_return_design_is_aligned() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.01, 200)
    design, target, scale = absolute_return_design(returns)
    assert design.shape[0] == returns.size
    assert np.all(scale > 0.0)
    assert np.isfinite(target[-2])


def test_windowed_batches_requires_complete_windows() -> None:
    design = np.ones((50, 2))
    target = np.ones(50)
    features, labels = windowed_batches(design, target, window=10)
    assert features.shape[1:] == (10, 2)
    assert labels.size == features.shape[0]


def test_expanding_drawdown_is_causal() -> None:
    prices = np.asarray([100.0, 90.0, 95.0, 120.0, 60.0])
    drawdown = expanding_drawdown(prices)
    assert drawdown[1] == pytest.approx(-0.1)
    assert drawdown[3] == pytest.approx(0.0)
    assert drawdown[4] == pytest.approx(-0.5)


def test_strata_partition_the_block() -> None:
    rng = np.random.default_rng(4)
    prices = np.cumprod(1.0 + rng.normal(0.0, 0.01, 300))
    variance = np.abs(rng.normal(1e-4, 1e-5, 300))
    labels = assign_strata(prices, variance)
    weights = stratum_weights(labels)
    assert float(weights.sum()) == pytest.approx(1.0)
    slices = stratum_slices(labels)
    assert sum(index.size for index in slices.values()) == labels.size


def test_standardiser_round_trip() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(5.0, 3.0, size=(100, 4))
    standardiser = Standardiser.fit(features)
    transformed = standardiser.transform(features)
    assert abs(float(transformed.mean())) < 1e-9
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)


def test_manifest_declares_https_sources() -> None:
    for spec in DATASETS.values():
        assert spec.url.startswith("https://")
    links = verified_links()
    assert len(links) >= 4
    assert availability() is not None


def test_aggregate_arm_respects_its_budget() -> None:
    rng = np.random.default_rng(6)
    from coupled_monotone_risk.data.credit import CreditPanel

    panels = [
        CreditPanel(
            name=f"panel{index}",
            features=rng.normal(size=(120, 6)),
            target=(rng.random(120) > 0.5).astype(np.int64),
            groups=np.arange(120, dtype=np.int64),
            week=np.full(120, -1, dtype=np.int64),
            prevalence=0.5,
        )
        for index in range(2)
    ]
    aggregate = build_aggregate(panels, budget=150)
    assert aggregate.size <= 150
    assert aggregate.name == "aggregate"
