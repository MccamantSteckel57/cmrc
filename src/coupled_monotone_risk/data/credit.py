"""Credit panels: German, Taiwan, Home Credit, Lending Club and the aggregate arm.

Ref: Sec. 4.1 (credit layer and the parallel aggregate arm), Table 1 Panel D.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from coupled_monotone_risk.data.manifest import DATASETS, MissingDataError, resolve
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class CreditPanel:
    name: str
    features: FloatArray
    target: IntArray
    groups: IntArray
    week: IntArray
    prevalence: float

    @property
    def size(self) -> int:
        return int(self.target.size)


@dataclass(frozen=True)
class Standardiser:
    centre: FloatArray
    scale: FloatArray
    fill: FloatArray

    @classmethod
    def fit(cls, features: FloatArray) -> Standardiser:
        matrix = np.asarray(features, dtype=np.float64)
        fill = np.nanmedian(matrix, axis=0)
        filled = np.where(np.isnan(matrix), fill, matrix)
        centre = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(centre=centre, scale=scale, fill=fill)

    def transform(self, features: FloatArray) -> FloatArray:
        matrix = np.asarray(features, dtype=np.float64)
        filled = np.where(np.isnan(matrix), self.fill, matrix)
        return np.asarray((filled - self.centre) / self.scale, dtype=np.float64)


def load_german(root: Path | None = None) -> CreditPanel:
    spec = DATASETS["credit_german"]
    numeric = (root or Path(".")) / spec.files[0] if root is not None else None
    path = numeric if numeric is not None and numeric.exists() else resolve(spec, root)
    if path.name.endswith("german.data-numeric"):
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        matrix = frame.iloc[:, :24].to_numpy(dtype=np.float64)
        labels = frame.iloc[:, 24].to_numpy()
    else:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        matrix = pd.get_dummies(frame.iloc[:, :20]).to_numpy(dtype=np.float64)
        labels = frame.iloc[:, 20].to_numpy()
    target = (np.asarray(labels) == 2).astype(np.int64)
    if matrix.shape[1] < 20:
        raise ValueError("German Credit file has fewer columns than documented")
    return CreditPanel(
        name="german",
        features=np.asarray(matrix, dtype=np.float64),
        target=np.asarray(target, dtype=np.int64),
        groups=np.arange(target.size, dtype=np.int64),
        week=np.full(target.size, -1, dtype=np.int64),
        prevalence=float(target.mean()),
    )


def load_taiwan(root: Path | None = None) -> CreditPanel:
    path = resolve(DATASETS["credit_taiwan"], root)
    frame = pd.read_excel(path, header=1)
    target_column = "default payment next month"
    if target_column not in frame.columns:
        raise ValueError("Taiwan panel is missing the default column")
    identifiers = frame["ID"].to_numpy(dtype=np.int64)
    target = frame[target_column].to_numpy(dtype=np.int64)
    features = frame.drop(columns=["ID", target_column]).to_numpy(dtype=np.float64)
    billing = [name for name in frame.columns if name.startswith("BILL_AMT")]
    payment = [name for name in frame.columns if name.startswith("PAY_AMT")]
    utilisation = (frame[billing].sum(axis=1) / frame["LIMIT_BAL"].clip(lower=1)).to_numpy()
    liquidity = (frame[payment].sum(axis=1) / frame["LIMIT_BAL"].clip(lower=1)).to_numpy()
    features = np.column_stack([features, utilisation, liquidity])
    return CreditPanel(
        name="taiwan",
        features=np.asarray(features, dtype=np.float64),
        target=np.asarray(target, dtype=np.int64),
        groups=np.asarray(identifiers, dtype=np.int64),
        week=np.full(target.size, -1, dtype=np.int64),
        prevalence=float(target.mean()),
    )


def load_home_credit(root: Path | None = None, max_rows: int | None = None) -> CreditPanel:
    path = resolve(DATASETS["credit_home"], root)
    frame = pd.read_csv(path, nrows=max_rows)
    if "TARGET" not in frame.columns or "SK_ID_CURR" not in frame.columns:
        raise ValueError("Home Credit panel is missing TARGET or SK_ID_CURR")
    week = frame["WEEK_NUM"].to_numpy(dtype=np.int64) if "WEEK_NUM" in frame else np.full(
        len(frame), -1, dtype=np.int64
    )
    drop = [name for name in ("SK_ID_CURR", "TARGET", "WEEK_NUM") if name in frame.columns]
    numeric = frame.drop(columns=drop).select_dtypes(include=[np.number])
    if numeric.shape[1] < 10:
        raise ValueError("Home Credit panel exposes fewer than ten numeric columns")
    return CreditPanel(
        name="home",
        features=np.asarray(numeric.to_numpy(dtype=np.float64), dtype=np.float64),
        target=np.asarray(frame["TARGET"].to_numpy(dtype=np.int64), dtype=np.int64),
        groups=np.asarray(frame["SK_ID_CURR"].to_numpy(dtype=np.int64), dtype=np.int64),
        week=week,
        prevalence=float(frame["TARGET"].mean()),
    )


def load_lending(root: Path | None = None, max_rows: int | None = None) -> CreditPanel:
    path = resolve(DATASETS["credit_lending"], root)
    frame = pd.read_csv(path, nrows=max_rows, low_memory=False)
    status_column = "loan_status"
    if status_column not in frame.columns:
        raise ValueError("Lending Club panel is missing loan_status")
    status = frame[status_column].astype(str).str.lower()
    target = status.isin(["charged off", "default", "late (31-120 days)"]).to_numpy(dtype=np.int64)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] < 5:
        raise ValueError("Lending Club panel exposes fewer than five numeric columns")
    return CreditPanel(
        name="lending",
        features=np.asarray(numeric.to_numpy(dtype=np.float64), dtype=np.float64),
        target=np.asarray(target, dtype=np.int64),
        groups=np.arange(target.size, dtype=np.int64),
        week=np.full(target.size, -1, dtype=np.int64),
        prevalence=float(np.mean(target)),
    )


def build_aggregate(panels: list[CreditPanel], budget: int = 300_000) -> CreditPanel:
    """Concatenate public panels on a shared standardised projection.

    Ref: Sec. 4.1 (aggregate arm, own compilation, never pooled with Home Credit).
    """
    if not panels:
        raise ValueError("aggregate arm requires at least one panel")
    width = min(panel.features.shape[1] for panel in panels)
    blocks: list[FloatArray] = []
    targets: list[IntArray] = []
    groups: list[IntArray] = []
    offset = 0
    for panel in panels:
        block = panel.features[:, :width]
        standardised = Standardiser.fit(block).transform(block)
        blocks.append(standardised)
        targets.append(panel.target)
        groups.append(panel.groups + offset)
        offset += int(panel.groups.max()) + 1
    features = np.vstack(blocks)
    target = np.concatenate(targets)
    group = np.concatenate(groups)
    if features.shape[0] > budget:
        stride = int(np.ceil(features.shape[0] / budget))
        keep = np.arange(0, features.shape[0], stride)[:budget]
        features, target, group = features[keep], target[keep], group[keep]
    LOGGER.info("aggregate arm rows=%d width=%d", features.shape[0], features.shape[1])
    return CreditPanel(
        name="aggregate",
        features=np.asarray(features, dtype=np.float64),
        target=np.asarray(target, dtype=np.int64),
        groups=np.asarray(group, dtype=np.int64),
        week=np.full(target.size, -1, dtype=np.int64),
        prevalence=float(target.mean()),
    )


def load_available(root: Path | None = None) -> dict[str, CreditPanel]:
    loaders = {
        "german": load_german,
        "taiwan": load_taiwan,
        "home": load_home_credit,
        "lending": load_lending,
    }
    panels: dict[str, CreditPanel] = {}
    for name, loader in loaders.items():
        try:
            panels[name] = loader(root)
        except (MissingDataError, FileNotFoundError) as error:
            LOGGER.info("credit panel %s unavailable: %s", name, error)
    return panels
