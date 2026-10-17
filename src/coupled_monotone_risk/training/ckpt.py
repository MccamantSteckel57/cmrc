"""Checkpoints with seed state and atomic replacement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from coupled_monotone_risk.utils.atomic import atomic_torch_load, atomic_torch_save
from coupled_monotone_risk.utils.seeding import set_seed

FORMAT_VERSION = 1


@dataclass(frozen=True)
class CheckpointMeta:
    epoch: int
    seed: int
    format_version: int = FORMAT_VERSION
    extra: dict[str, Any] | None = None


def save_checkpoint(
    path: Path,
    module: nn.Module,
    meta: CheckpointMeta,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "format_version": meta.format_version,
        "epoch": meta.epoch,
        "seed": meta.seed,
        "state_dict": module.state_dict(),
        "config": dict(meta.extra or {}),
    }
    if optimiser is not None:
        payload["optimiser"] = optimiser.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    return atomic_torch_save(path, payload)


def load_checkpoint(
    path: Path,
    module: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_seed: bool = True,
) -> CheckpointMeta:
    payload = atomic_torch_load(path)
    version = int(payload.get("format_version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {version}")
    module.load_state_dict(payload["state_dict"])
    if optimiser is not None and "optimiser" in payload:
        optimiser.load_state_dict(payload["optimiser"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    seed = int(payload["seed"])
    if restore_seed:
        set_seed(seed)
    return CheckpointMeta(
        epoch=int(payload["epoch"]),
        seed=seed,
        format_version=version,
        extra=dict(payload.get("config", {})),
    )
