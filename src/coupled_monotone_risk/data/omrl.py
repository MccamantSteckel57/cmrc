"""Oxford-Man realized-library reader for the 21-index subset.

Ref: Sec. 4.1 (OMRL-21, frozen snapshot 2022-06-28).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coupled_monotone_risk.data.manifest import DATASETS, MissingDataError, resolve
from coupled_monotone_risk.data.regimes import parse_day
from coupled_monotone_risk.data.series import DailySeries
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_SUBSET: tuple[str, ...] = (
    ".AEX",
    ".BFX",
    ".BSESN",
    ".BVLG",
    ".BVSP",
    ".FCHI",
    ".FTMIB",
    ".FTSE",
    ".GDAXI",
    ".GSPTSE",
    ".HSI",
    ".IBEX",
    ".IPSAP",
    ".ISEQ",
    ".JKSE",
    ".KLSE",
    ".KS11",
    ".MXX",
    ".N225",
    ".OMXC20",
    ".OMXS30",
)

REALIZED_COLUMNS: tuple[str, ...] = ("rv5", "rv", "RV5", "RV", "realized_variance")


@dataclass(frozen=True)
class ArchiveTable:
    frame: pd.DataFrame
    symbol: str

    @property
    def size(self) -> int:
        return int(self.frame.shape[0])


def _date_column(frame: pd.DataFrame) -> str:
    for candidate in ("Date", "date", "DATE"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("realized-library table has no date column")


def _realized_column(frame: pd.DataFrame) -> str:
    for candidate in REALIZED_COLUMNS:
        if candidate in frame.columns:
            return candidate
    raise ValueError("realized-library table has no realized-variance column")


def read_archive_table(path: Path, symbol: str | None = None) -> ArchiveTable:
    frame = pd.read_csv(path)
    _date_column(frame)
    _realized_column(frame)
    if "Close" not in frame.columns and "close" not in frame.columns:
        raise ValueError("realized-library table has no close column")
    resolved = symbol or path.stem
    return ArchiveTable(frame=frame, symbol=resolved)


def to_daily_series(table: ArchiveTable) -> DailySeries:
    frame = table.frame.copy()
    frame["_day"] = [parse_day(str(value)[:10]) for value in frame[_date_column(frame)]]
    frame = frame.sort_values("_day").drop_duplicates("_day")
    close_column = "Close" if "Close" in frame.columns else "close"
    variance = frame[_realized_column(frame)].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError(f"{table.symbol}: realized variance must be finite and positive")
    return DailySeries(
        symbol=table.symbol,
        days=tuple(frame["_day"].tolist()),
        close=np.asarray(frame[close_column].to_numpy(dtype=np.float64), dtype=np.float64),
        realized_variance=np.asarray(variance, dtype=np.float64),
    )


def load_asset(symbol: str, root: Path | None = None) -> DailySeries:
    spec = DATASETS["omrl21"]
    base = resolve(spec, root)
    candidates = [base.parent / f"{symbol}.csv", base.parent / f"{symbol.lstrip('.')}.csv"]
    for candidate in candidates:
        if candidate.exists():
            return to_daily_series(read_archive_table(candidate, symbol))
    raise FileNotFoundError(f"no archive file for {symbol} beside {base.name}")


def load_subset(
    root: Path | None = None, subset: tuple[str, ...] = DEFAULT_SUBSET
) -> dict[str, DailySeries]:
    series: dict[str, DailySeries] = {}
    for symbol in subset:
        try:
            series[symbol] = load_asset(symbol, root)
        except (MissingDataError, FileNotFoundError):
            LOGGER.info("realized-library asset %s not present", symbol)
    return series
