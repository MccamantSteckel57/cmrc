"""Daily equity-bar reader for the return-interval layer.

Ref: Sec. 4.1 (frozen Amazon and Alphabet daily series).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coupled_monotone_risk.data.manifest import DATASETS, MissingDataError, resolve
from coupled_monotone_risk.data.regimes import parse_day
from coupled_monotone_risk.data.series import DailySeries

_DATE_KEYS = ("Date", "date", "DATE", "timestamp")
_CLOSE_KEYS = ("Close", "close", "CLOSE", "Adj Close", "adj_close")


def _column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"equity table has no {label} column")


def load_equity(key: str, root: Path | None = None) -> DailySeries:
    if key not in ("equity_amzn", "equity_googl"):
        raise KeyError(f"{key!r} is not an equity source")
    path = resolve(DATASETS[key], root)
    frame = pd.read_csv(path)
    date_column = _column(frame, _DATE_KEYS, "date")
    close_column = _column(frame, _CLOSE_KEYS, "close")
    records = {
        parse_day(str(value)[:10]): float(price)
        for value, price in zip(frame[date_column], frame[close_column], strict=True)
    }
    days = sorted(records)
    close = np.asarray([records[day] for day in days], dtype=np.float64)
    if close.size < 30:
        raise ValueError(f"{key}: fewer than thirty usable bars")
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    variance = np.asarray(returns**2, dtype=np.float64)
    symbol = key.replace("equity_", "").upper()
    return DailySeries(symbol=symbol, days=tuple(days), close=close, realized_variance=variance)


def load_pair(root: Path | None = None) -> dict[str, DailySeries]:
    pair: dict[str, DailySeries] = {}
    for key in ("equity_amzn", "equity_googl"):
        try:
            pair[key.replace("equity_", "").upper()] = load_equity(key, root)
        except (MissingDataError, FileNotFoundError, ValueError):
            continue
    return pair
