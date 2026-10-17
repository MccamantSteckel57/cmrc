"""Chronological block layout with embargo (Sec. 3.1, Sec. 4.6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BlockLayout:
    length: int
    train_stop: int
    embargo_stop: int
    calib_stop: int
    embargo: int = field(default=22)

    def __post_init__(self) -> None:
        if not 0 < self.train_stop < self.embargo_stop < self.calib_stop <= self.length:
            raise ValueError("blocks must satisfy 0 < t1 < t2 < t3 <= N")
        if self.embargo_stop - self.train_stop != self.embargo:
            raise ValueError("embargo block length does not match the embargo setting")

    def train(self) -> npt.NDArray[np.int64]:
        return np.arange(0, self.train_stop, dtype=np.int64)

    def embargo_block(self) -> npt.NDArray[np.int64]:
        return np.arange(self.train_stop, self.embargo_stop, dtype=np.int64)

    def calibration(self) -> npt.NDArray[np.int64]:
        return np.arange(self.embargo_stop, self.calib_stop, dtype=np.int64)

    def test(self) -> npt.NDArray[np.int64]:
        return np.arange(self.calib_stop, self.length, dtype=np.int64)

    @property
    def calibration_size(self) -> int:
        return self.calib_stop - self.embargo_stop

    @property
    def test_size(self) -> int:
        return self.length - self.calib_stop


def plan_blocks(length: int, train_fraction: float, calib_fraction: float, embargo: int) -> BlockLayout:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie inside (0, 1)")
    if not 0.0 < calib_fraction < 1.0 - train_fraction:
        raise ValueError("calibration fraction leaves no test block")
    train_stop = round(length * train_fraction)
    calib_stop = round(length * (train_fraction + calib_fraction))
    if train_stop <= 0 or calib_stop >= length:
        raise ValueError("block layout leaves an empty segment")
    return BlockLayout(
        length=length,
        train_stop=train_stop,
        embargo_stop=train_stop + embargo,
        calib_stop=calib_stop,
        embargo=embargo,
    )


@dataclass(frozen=True)
class RollingWindow:
    calibration_start: int
    calibration_stop: int
    test_start: int
    test_stop: int

    @property
    def calibration_size(self) -> int:
        return self.calibration_stop - self.calibration_start

    @property
    def test_size(self) -> int:
        return self.test_stop - self.test_start


def rolling_windows(
    length: int, window: int = 250, embargo: int = 22, step: int = 1, horizon: int = 1
) -> list[RollingWindow]:
    if window <= embargo + 1:
        raise ValueError("rolling window must exceed the embargo")
    windows: list[RollingWindow] = []
    cursor = window
    while cursor + horizon <= length:
        windows.append(
            RollingWindow(
                calibration_start=cursor - window,
                calibration_stop=cursor - embargo,
                test_start=cursor,
                test_stop=min(cursor + horizon, length),
            )
        )
        cursor += step
    return windows


def slice_series(values: npt.ArrayLike, indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(array[indices], dtype=np.float64)
