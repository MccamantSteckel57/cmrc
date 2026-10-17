"""Learning-rate schedules."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimiser: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float = 0.0,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must lie inside [0, 1]")
    warmup = min(warmup_steps, total_steps)

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(warmup, 1))
        progress = (step - warmup) / float(max(total_steps - warmup, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return float(minimum_ratio + (1.0 - minimum_ratio) * cosine)

    return LambdaLR(optimiser, lr_lambda=factor)


def linear_warmup(optimiser: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    def factor(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        return float(min(step + 1, warmup_steps)) / float(warmup_steps)

    return LambdaLR(optimiser, lr_lambda=factor)
