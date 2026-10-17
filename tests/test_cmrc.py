import numpy as np
import pytest

from coupled_monotone_risk.calibration.cmrc import CmrcCalibrator, TaskInput
from coupled_monotone_risk.risks.family import IntervalFamily, SetFamily


def test_calibration_is_feasible_and_monotone(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    calibrator = CmrcCalibrator(budget=0.1, grid_size=401)
    result = calibrator.fit([TaskInput("a", first, 0.5), TaskInput("b", second, 0.5)])
    assert result.feasible
    assert result.diagnostics.non_monotone_share == 0.0
    assert result.diagnostics.monotonicity_share == 1.0
    assert result.achieved_risk <= 0.1 + 1e-9


def test_certificates_carry_the_allocated_budget(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    result = CmrcCalibrator(budget=0.1, grid_size=301).fit(
        [TaskInput("a", first, 0.5), TaskInput("b", second, 0.5)]
    )
    for certificate in result.tasks:
        assert certificate.calibrated_risk <= certificate.budget + 0.02
        assert certificate.width > 0.0


def test_shares_sum_to_one_under_weights(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    result = CmrcCalibrator(budget=0.1, grid_size=301).fit(
        [TaskInput("a", first, 0.7), TaskInput("b", second, 0.3)]
    )
    total = sum(task.weight * task.share for task in result.tasks)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_mixed_task_types_calibrate_together(
    mixed_pair: tuple[IntervalFamily, SetFamily],
) -> None:
    interval, labels = mixed_pair
    result = CmrcCalibrator(budget=0.1, grid_size=301).fit(
        [TaskInput("interval", interval, 0.5), TaskInput("set", labels, 0.5)]
    )
    assert result.feasible
    assert len(result.tasks) == 2
    assert result.tasks[1].width >= 1.0


def test_stratified_calibration_records_weights(
    mixed_pair: tuple[IntervalFamily, SetFamily],
) -> None:
    interval, _ = mixed_pair
    strata = np.asarray([0] * 200 + [1] * 150 + [2] * 150, dtype=np.int64)
    result = CmrcCalibrator(budget=0.1, grid_size=301).fit(
        [TaskInput("interval", interval, 1.0)], [strata]
    )
    assert result.diagnostics.stratum_weights.size == 3
    assert float(result.diagnostics.stratum_weights.sum()) == pytest.approx(1.0)


def test_calibration_is_deterministic(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    tasks = [TaskInput("a", first, 0.5), TaskInput("b", second, 0.5)]
    left = CmrcCalibrator(budget=0.1, grid_size=301).fit(tasks)
    right = CmrcCalibrator(budget=0.1, grid_size=301).fit(tasks)
    assert left.level == right.level
    assert [task.threshold for task in left.tasks] == [task.threshold for task in right.tasks]


def test_band_raises_the_certificate_level(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, _ = interval_pair
    plain = CmrcCalibrator(budget=0.1, grid_size=301).fit([TaskInput("a", first, 1.0)])
    banded = CmrcCalibrator(budget=0.1, grid_size=301, shift_band=0.02).fit(
        [TaskInput("a", first, 1.0)]
    )
    assert banded.certificate_budget() == pytest.approx(0.12)
    assert banded.diagnostics.shift_band >= plain.diagnostics.shift_band


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        CmrcCalibrator(budget=1.2)


def test_fixed_shares_are_respected(
    interval_pair: tuple[IntervalFamily, IntervalFamily],
) -> None:
    first, second = interval_pair
    result = CmrcCalibrator(budget=0.1, grid_size=301).fit(
        [TaskInput("a", first, 0.5), TaskInput("b", second, 0.5)],
        shares=np.asarray([1.0, 1.0]),
    )
    assert [task.budget for task in result.tasks] == pytest.approx([0.1, 0.1])
