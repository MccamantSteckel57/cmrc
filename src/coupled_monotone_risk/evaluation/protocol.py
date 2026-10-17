"""The evaluation grid: task x asset x regime, with per-regime summaries.

Ref: Sec. 4.1 (protocol), Sec. 4.6 (regime table), Figure 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.metrics.interval import wald_interval
from coupled_monotone_risk.risks.family import NestedFamily
from coupled_monotone_risk.training.trainer import TrainConfig
from coupled_monotone_risk.utils.numeric import FloatArray

IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]

UNDERPOWERED_THRESHOLD = 30


@dataclass(frozen=True)
class ProtocolConfig:
    budget: float = 0.10
    window: int = 250
    embargo: int = 22
    train_fraction: float = 0.60
    calib_fraction: float = 0.20
    grid_size: int = 1001
    delta: float = 0.05
    shift_band: float = 0.0
    seed: int = 13
    strata: int = 3
    horizons: tuple[int, ...] = (1,)
    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    hidden: int = 64

    def __post_init__(self) -> None:
        if not 0.0 < self.budget < 1.0:
            raise ValueError("the budget must lie inside (0, 1)")
        if self.window <= self.embargo:
            raise ValueError("the rolling window must exceed the embargo")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("the training fraction must lie inside (0, 1)")
        if not 0.0 < self.calib_fraction < 1.0 - self.train_fraction:
            raise ValueError("the calibration fraction leaves no test block")

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            seed=self.seed,
            log_every=0,
        )


@dataclass(frozen=True)
class IntervalTask:
    name: str
    asset: str
    family: NestedFamily
    point: FloatArray
    truth: FloatArray
    scale: FloatArray
    regimes: StrArray
    kind: str = "interval"

    def __post_init__(self) -> None:
        if not (self.point.size == self.truth.size == self.scale.size == self.regimes.size):
            raise ValueError("interval task arrays must be aligned")

    def hits(self, threshold: float) -> IntArray:
        residual = np.abs(self.truth - self.point)
        return np.asarray((residual <= threshold * self.scale).astype(np.int64), dtype=np.int64)

    def widths(self, threshold: float) -> FloatArray:
        return np.asarray(2.0 * threshold * self.scale, dtype=np.float64)


@dataclass(frozen=True)
class SetTask:
    name: str
    asset: str
    family: NestedFamily
    probabilities: FloatArray
    truth: IntArray
    regimes: StrArray
    kind: str = "set"

    def __post_init__(self) -> None:
        if self.probabilities.shape[0] != self.truth.size:
            raise ValueError("probability rows must match the label count")

    def hits(self, threshold: float) -> IntArray:
        included = self.probabilities >= (1.0 - threshold)
        selected = included[np.arange(self.truth.size), self.truth]
        return np.asarray(selected.astype(np.int64), dtype=np.int64)

    def widths(self, threshold: float) -> FloatArray:
        included = self.probabilities >= (1.0 - threshold)
        return np.asarray(np.maximum(included.sum(axis=1), 1).astype(np.float64), dtype=np.float64)


Task = IntervalTask | SetTask


@dataclass(frozen=True)
class RegimeCell:
    regime: str
    n: int
    coverage: float
    width: float
    gap_pp: float
    wald_low: float
    wald_high: float
    underpowered: bool

    @property
    def exceeds_budget(self) -> bool:
        return self.gap_pp > 100.0 * 0.05


@dataclass(frozen=True)
class ArmResult:
    arm: str
    task: str
    kind: str
    asset: str
    level: float
    budget: float
    cells: tuple[RegimeCell, ...]
    joint_coverage: float
    joint_width: float
    worst_gap_pp: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def cell(self, regime: str) -> RegimeCell:
        for entry in self.cells:
            if entry.regime == regime:
                return entry
        raise KeyError(f"regime {regime!r} is absent from arm {self.arm!r}")


def summarise(task: Task, arm: str, threshold: float, level: float, budget: float) -> ArmResult:
    hits = task.hits(threshold)
    widths = task.widths(threshold)
    cells: list[RegimeCell] = []
    for regime in np.unique(task.regimes):
        mask = task.regimes == regime
        count = int(mask.sum())
        if count == 0:
            continue
        coverage = float(hits[mask].mean())
        low, high = wald_interval(hits[mask])
        cells.append(
            RegimeCell(
                regime=str(regime),
                n=count,
                coverage=coverage,
                width=float(widths[mask].mean()),
                gap_pp=float(100.0 * abs(coverage - (1.0 - budget))),
                wald_low=low,
                wald_high=high,
                underpowered=count < UNDERPOWERED_THRESHOLD,
            )
        )
    gaps = [cell.gap_pp for cell in cells if cell.regime != "R1"]
    return ArmResult(
        arm=arm,
        task=task.name,
        kind=task.kind,
        asset=task.asset,
        level=float(level),
        budget=float(budget),
        cells=tuple(cells),
        joint_coverage=float(hits.mean()),
        joint_width=float(widths.mean()),
        worst_gap_pp=float(max(gaps)) if gaps else float("nan"),
    )
