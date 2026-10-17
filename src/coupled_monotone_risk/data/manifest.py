"""Dataset registry, on-disk resolution and provenance records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class MissingDataError(RuntimeError):
    """Raised when a declared source is not present in the local data root."""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    panel: str
    url: str
    license: str
    version: str
    files: tuple[str, ...]
    access: str
    note: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "omrl21": DatasetSpec(
        key="omrl21",
        name="Oxford-Man Realized Library, 21-index subset",
        panel="volatility",
        url="https://www.oxford-man.ox.ac.uk/research/realized-library/",
        license="Oxford-Man Institute realized-library terms of use",
        version="frozen snapshot 2022-06-28",
        files=("omrl/RealizedLibrary.csv",),
        access="public",
        note="archive closed for download; the curated archive copy is required",
    ),
    "equity_amzn": DatasetSpec(
        key="equity_amzn",
        name="Amazon daily bars",
        panel="equity",
        url="https://stooq.com/q/d/?s=amzn.us",
        license="end-of-day quote terms of the distributing venue",
        version="snapshot frozen before evaluation",
        files=("equity/amzn.csv",),
        access="public",
    ),
    "equity_googl": DatasetSpec(
        key="equity_googl",
        name="Alphabet daily bars",
        panel="equity",
        url="https://stooq.com/q/d/?s=googl.us",
        license="end-of-day quote terms of the distributing venue",
        version="snapshot frozen before evaluation",
        files=("equity/googl.csv",),
        access="public",
    ),
    "credit_german": DatasetSpec(
        key="credit_german",
        name="Statlog German Credit",
        panel="credit",
        url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        license="CC BY 4.0",
        version="UCI revision 2023-06-09",
        files=("credit/german.data-numeric", "credit/german.data"),
        access="public",
    ),
    "credit_taiwan": DatasetSpec(
        key="credit_taiwan",
        name="Default of Credit Card Clients",
        panel="credit",
        url="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        license="CC BY 4.0",
        version="UCI revision 2023-05-22",
        files=("credit/default of credit card clients.xls",),
        access="public",
    ),
    "credit_home": DatasetSpec(
        key="credit_home",
        name="Home Credit Default Risk application panel",
        panel="credit",
        url="https://www.kaggle.com/competitions/home-credit-default-risk/data",
        license="Home Credit competition rules",
        version="competition release",
        files=("credit/application_train.csv",),
        access="credentialed",
    ),
    "credit_lending": DatasetSpec(
        key="credit_lending",
        name="Lending Club loan panel",
        panel="credit",
        url="https://www.lendingclub.com/info/download-data.action",
        license="Lending Club data-use terms",
        version="issued-loan extract",
        files=("credit/loan.csv",),
        access="credentialed",
    ),
    "fi2010": DatasetSpec(
        key="fi2010",
        name="FI-2010 limit-order-book benchmark",
        panel="orderbook",
        url="https://etsin.fairdata.fi/dataset/73eb48b4-a9bc-4b6f-9ade-9c8b8a0d1f0e",
        license="CC BY 4.0",
        version="benchmark release",
        files=("fi2010/data.csv",),
        access="public",
        note="ten days only, no regime statement is supported",
    ),
}


def data_root() -> Path:
    return Path(os.environ.get("CMRC_DATA_ROOT", "data"))


def resolve(spec: DatasetSpec, root: Path | None = None) -> Path:
    base = data_root() if root is None else root
    for relative in spec.files:
        candidate = base / relative
        if candidate.exists():
            return candidate
    raise MissingDataError(
        f"{spec.name}: none of {', '.join(spec.files)} is present under the data root"
    )


def availability(root: Path | None = None) -> dict[str, str]:
    base = data_root() if root is None else root
    status: dict[str, str] = {}
    for key, spec in DATASETS.items():
        present = any((base / relative).exists() for relative in spec.files)
        status[key] = "PRESENT" if present else "ABSENT"
    return status


def verified_links() -> list[tuple[str, str]]:
    return [(spec.name, spec.url) for spec in DATASETS.values() if spec.access == "public"]
