"""Optimiser and parameter-group construction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import nn

OptimiserName = Literal["adamw", "adam", "sgd"]


def decay_groups(module: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimiser(
    parameters: Iterable[nn.Parameter] | list[dict[str, object]],
    name: OptimiserName = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    if lr <= 0.0:
        raise ValueError("the learning rate must be strictly positive")
    grouped: list[Any] = list(parameters)
    if name == "adamw":
        return torch.optim.AdamW(grouped, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(grouped, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            grouped, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"unknown optimiser {name!r}")
