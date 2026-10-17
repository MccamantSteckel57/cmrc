"""Shared in-memory representation of a daily financial series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DailySeries:
    symbol: str
    days: tuple[date, ...]
    close: FloatArray
    realized_variance: FloatArray

    def __post_init__(self) -> None:
        if not (len(self.days) == self.close.size == self.realized_variance.size):
            raise ValueError("days, close and realized variance must have equal length")
        if self.close.size < 30:
            raise ValueError("a daily series needs at least thirty observations")
        if np.any(self.close <= 0.0):
            raise ValueError("close prices must be strictly positive")

    @property
    def size(self) -> int:
        return int(self.close.size)

    def log_returns(self) -> FloatArray:
        log_close = np.log(as_float_array(self.close))
        returns = np.diff(log_close, prepend=log_close[0])
        returns[0] = 0.0
        return np.asarray(returns, dtype=np.float64)

    def subset(self, start: int, stop: int) -> DailySeries:
        return DailySeries(
            symbol=self.symbol,
            days=self.days[start:stop],
            close=self.close[start:stop],
            realized_variance=self.realized_variance[start:stop],
        )

    def overlapping_days(self, other: DailySeries) -> IntArray:
        index = {day: position for position, day in enumerate(other.days)}
        positions = [index[day] for day in self.days if day in index]
        return np.asarray(positions, dtype=np.int64)
