"""Frozen evaluation taxonomy of seven market windows (Sec. 4.1, Table 4.6).

The taxonomy labels results after they are produced. It never enters calibration,
never indexes a stratum and never serves as a predictor (Remark 1).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RegimeWindow:
    label: str
    name: str
    start: date
    end: date
    drawdown: float
    stressed: bool
    underpowered: bool
    note: str

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


REGIME_WINDOWS: tuple[RegimeWindow, ...] = (
    RegimeWindow(
        "R1",
        "calm bull",
        date(2004, 1, 2),
        date(2007, 10, 8),
        0.0,
        False,
        False,
        "calm reference, excluded from the worst-case summary",
    ),
    RegimeWindow(
        "R2",
        "global financial crisis",
        date(2007, 10, 9),
        date(2009, 3, 9),
        -0.568,
        True,
        False,
        "peak-to-trough drawdown",
    ),
    RegimeWindow(
        "R3",
        "post-crisis expansion",
        date(2009, 3, 10),
        date(2020, 2, 18),
        0.0,
        False,
        False,
        "long expansion",
    ),
    RegimeWindow(
        "R4",
        "COVID crash",
        date(2020, 2, 19),
        date(2020, 3, 23),
        -0.339,
        True,
        True,
        "33 calendar days, about 23 observations",
    ),
    RegimeWindow(
        "R5",
        "post-COVID reflation",
        date(2020, 3, 24),
        date(2022, 1, 2),
        0.0,
        False,
        False,
        "reflation",
    ),
    RegimeWindow(
        "R6",
        "rate-hike bear",
        date(2022, 1, 3),
        date(2022, 10, 12),
        -0.254,
        True,
        False,
        "slow bleed over 282 calendar days",
    ),
    RegimeWindow(
        "R7",
        "post-stress recovery",
        date(2022, 10, 13),
        date(2026, 10, 9),
        0.0,
        False,
        False,
        "contains the two 2024-2026 corrections",
    ),
)

PUBLICATION_COUNTS: dict[str, int] = {
    "R1": 949,
    "R2": 356,
    "R3": 2760,
    "R4": 23,
    "R5": 448,
    "R6": 193,
    "R7": 1006,
}


def parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def label_for(day: date, windows: Iterable[RegimeWindow] = REGIME_WINDOWS) -> str:
    for window in windows:
        if window.contains(day):
            return window.label
    raise ValueError(f"{day.isoformat()} falls outside the frozen taxonomy")


def window_of(label: str) -> RegimeWindow:
    for window in REGIME_WINDOWS:
        if window.label == label:
            return window
    raise KeyError(f"unknown regime label {label!r}")


def label_series(days: Iterable[date]) -> npt.NDArray[np.str_]:
    return np.asarray([label_for(day) for day in days], dtype=np.str_)


def coverage_report(days: Iterable[date]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for day in days:
        label = label_for(day)
        counts[label] = counts.get(label, 0) + 1
    return counts


def taxonomy_sums_to_published_total() -> bool:
    return sum(PUBLICATION_COUNTS.values()) == 5735
