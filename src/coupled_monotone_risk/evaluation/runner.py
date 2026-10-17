"""Arm construction and execution over the protocol grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from coupled_monotone_risk.calibration.cmrc import CalibrationResult, CmrcCalibrator, TaskInput
from coupled_monotone_risk.data.credit import CreditPanel, Standardiser
from coupled_monotone_risk.data.features import (
    finite_rows,
    har_design,
    har_log_design,
    trailing_mean,
)
from coupled_monotone_risk.data.regimes import label_series
from coupled_monotone_risk.data.series import DailySeries
from coupled_monotone_risk.data.splits import plan_blocks
from coupled_monotone_risk.evaluation.protocol import (
    ArmResult,
    IntervalTask,
    ProtocolConfig,
    SetTask,
    Task,
    summarise,
)
from coupled_monotone_risk.models import registry
from coupled_monotone_risk.models.credit import CreditModel
from coupled_monotone_risk.risks.family import IntervalFamily, SetFamily
from coupled_monotone_risk.training.trainer import TrainingReport
from coupled_monotone_risk.utils.numeric import FloatArray
from coupled_monotone_risk.utils.seeding import set_seed


@dataclass(frozen=True)
class Prepared:
    task_input: TaskInput
    task: Task
    calib_labels: np.ndarray | None
    test_labels: np.ndarray | None
    meta: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyRun:
    calibration: CalibrationResult
    arms: tuple[ArmResult, ...]
    reports: tuple[TrainingReport, ...]
    prepared: tuple[Prepared, ...]


def _split(size: int, config: ProtocolConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = plan_blocks(size, config.train_fraction, config.calib_fraction, config.embargo)
    return layout.train(), layout.calibration(), layout.test()


def _credit_regimes(panel: CreditPanel, indices: np.ndarray) -> np.ndarray:
    if panel.week.size == panel.target.size and np.all(panel.week[indices] >= 0):
        weeks = panel.week[indices]
        edges = np.quantile(weeks, [0.25, 0.5, 0.75])
        bucket = np.searchsorted(edges, weeks, side="right")
        return np.asarray([f"W{int(value) + 1}" for value in bucket], dtype=np.str_)
    return np.asarray(["SINGLE"] * indices.size, dtype=np.str_)


def prepare_credit_arm(
    panel: CreditPanel, model_name: str, config: ProtocolConfig, weight: float = 1.0
) -> tuple[Prepared, TrainingReport]:
    set_seed(config.seed)
    train, calibration, test = _split(panel.size, config)
    standardiser = Standardiser.fit(panel.features[train])
    features = standardiser.transform(panel.features)
    model = registry.build(
        model_name,
        {
            "width": int(features.shape[1]),
            "hidden": config.hidden,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "log_every": 0,
        },
    )
    if not isinstance(model, CreditModel):
        raise TypeError("the credit arm needs a credit scorer")
    report = model.fit(features[train], panel.target[train])
    calibration_probabilities = model.predict_proba(features[calibration])
    test_probabilities = model.predict_proba(features[test])
    family = SetFamily(calibration_probabilities, panel.target[calibration])
    task = SetTask(
        name=panel.name,
        asset=panel.name,
        family=family,
        probabilities=test_probabilities,
        truth=panel.target[test],
        regimes=_credit_regimes(panel, test),
    )
    prepared = Prepared(
        task_input=TaskInput(name=panel.name, family=family, weight=weight),
        task=task,
        calib_labels=None,
        test_labels=None,
        meta={
            "prevalence": panel.prevalence,
            "train": float(train.size),
            "calibration": float(calibration.size),
            "test": float(test.size),
        },
    )
    return prepared, report


def _volatility_design(series: DailySeries, log_target: bool) -> tuple[FloatArray, FloatArray]:
    if log_target:
        return har_log_design(series.realized_variance, horizon=1)
    return har_design(series.realized_variance, horizon=1)


def prepare_volatility_arm(
    series: DailySeries, model_name: str, config: ProtocolConfig, weight: float = 1.0
) -> tuple[Prepared, TrainingReport | None]:
    set_seed(config.seed)
    log_target = model_name == "har_log"
    design, target = _volatility_design(series, log_target)
    usable = finite_rows(design, target)
    if usable.size < 120:
        raise ValueError(f"{series.symbol}: too few complete HAR rows")
    train, calibration, test = _split(usable.size, config)
    train_rows, calib_rows, test_rows = usable[train], usable[calibration], usable[test]
    model = registry.build(model_name, {"seed": config.seed})
    model.fit(design[train_rows], target[train_rows])
    fitted = model.predict(design)
    point_calib = np.asarray(fitted[calib_rows], dtype=np.float64)
    point_test = np.asarray(fitted[test_rows], dtype=np.float64)
    truth_calib = np.asarray(target[calib_rows], dtype=np.float64)
    truth_test = np.asarray(target[test_rows], dtype=np.float64)
    scale = trailing_mean(series.realized_variance, 22)
    scale_calib = np.nan_to_num(scale[calib_rows], nan=float(np.nanmean(series.realized_variance)))
    scale_test = np.nan_to_num(scale[test_rows], nan=float(np.nanmean(series.realized_variance)))
    family = IntervalFamily(np.abs(truth_calib - point_calib) / scale_calib, float(np.mean(scale_test)))
    regimes = label_series([series.days[index] for index in test_rows])
    task = IntervalTask(
        name=series.symbol,
        asset=series.symbol,
        family=family,
        point=point_test,
        truth=truth_test,
        scale=scale_test,
        regimes=regimes,
    )
    prepared = Prepared(
        task_input=TaskInput(name=series.symbol, family=family, weight=weight),
        task=task,
        calib_labels=None,
        test_labels=None,
        meta={"rows": float(usable.size)},
    )
    return prepared, getattr(model, "report", None)


def prepare_return_arm(
    series: DailySeries, model_name: str, config: ProtocolConfig, weight: float = 1.0
) -> tuple[Prepared, TrainingReport | None]:
    set_seed(config.seed)
    returns = series.log_returns()
    if model_name == "ar":
        model = registry.build("ar", {"lags": 5})
        model.fit(returns[: int(returns.size * config.train_fraction)])
        fitted = model.filter(returns)
        report = None
    elif model_name == "transformer":
        context = 32
        windows, targets = _return_windows(returns, context)
        if windows.shape[0] < 200:
            raise ValueError(f"{series.symbol}: too few return windows")
        split = int(windows.shape[0] * config.train_fraction)
        model = registry.build(
            "transformer",
            {
                "context": context,
                "d_model": 32,
                "heads": 4,
                "layers": 2,
                "epochs": 30,
                "batch_size": 128,
                "lr": 1e-3,
                "seed": config.seed,
            },
        )
        report = model.fit(windows[:split], targets[:split])
        predicted = model.predict(windows)
        fitted = np.full(returns.size, np.nan, dtype=np.float64)
        fitted[context : context + predicted.size] = predicted
    else:
        raise KeyError(f"{model_name!r} is not a return-layer model")
    valid = np.flatnonzero(np.isfinite(fitted))
    if valid.size < 150:
        raise ValueError(f"{series.symbol}: too few finite return forecasts")
    _, calibration, test = _split(valid.size, config)
    calib_rows, test_rows = valid[calibration], valid[test]
    volatility = trailing_mean(np.abs(returns), 22)
    fallback = float(np.nanmean(np.abs(returns)))
    scale_calib = np.nan_to_num(volatility[calib_rows], nan=fallback)
    scale_test = np.nan_to_num(volatility[test_rows], nan=fallback)
    family = IntervalFamily(
        np.abs(returns[calib_rows] - fitted[calib_rows]) / scale_calib, float(np.mean(scale_test))
    )
    regimes = label_series([series.days[index] for index in test_rows])
    task = IntervalTask(
        name=series.symbol,
        asset=series.symbol,
        family=family,
        point=np.asarray(fitted[test_rows], dtype=np.float64),
        truth=np.asarray(returns[test_rows], dtype=np.float64),
        scale=scale_test,
        regimes=regimes,
    )
    prepared = Prepared(
        task_input=TaskInput(name=series.symbol, family=family, weight=weight),
        task=task,
        calib_labels=None,
        test_labels=None,
        meta={"rows": float(valid.size)},
    )
    return prepared, report


def _return_windows(returns: FloatArray, context: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=np.float64)
    windows = []
    targets = []
    for index in range(context, values.size):
        window = values[index - context : index]
        if np.all(np.isfinite(window)) and np.isfinite(values[index]):
            windows.append(window[:, None])
            targets.append(values[index])
    if not windows:
        raise ValueError("no complete return windows")
    return np.asarray(windows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def make_calibrator(config: ProtocolConfig) -> CmrcCalibrator:
    return CmrcCalibrator(
        budget=config.budget,
        grid_size=config.grid_size,
        shift_band=config.shift_band,
        strata=config.strata,
    )


def run_study(
    prepared: Sequence[Prepared],
    config: ProtocolConfig,
    arm: str = "cmrc",
    reports: Sequence[TrainingReport] | None = None,
) -> StudyRun:
    if not prepared:
        raise ValueError("a study needs at least one prepared task")
    calibrator = make_calibrator(config)
    calibration = calibrator.fit([entry.task_input for entry in prepared])
    arms = tuple(
        summarise(
            entry.task,
            arm=arm,
            threshold=certificate.threshold,
            level=calibration.level,
            budget=config.budget,
        )
        for entry, certificate in zip(prepared, calibration.tasks, strict=True)
    )
    return StudyRun(
        calibration=calibration,
        arms=arms,
        reports=tuple(reports or ()),
        prepared=tuple(prepared),
    )
