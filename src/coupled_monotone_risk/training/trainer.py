"""Supervised training loop shared by the volumetric and credit backbones."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from coupled_monotone_risk.training.optim import build_optimiser, decay_groups
from coupled_monotone_risk.training.sched import cosine_with_warmup
from coupled_monotone_risk.utils.logging import get_logger
from coupled_monotone_risk.utils.seeding import set_seed

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    seed: int = 13
    device: str = "cpu"
    amp: bool = False
    log_every: int = 20

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie inside [0, 1)")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingReport:
    history: tuple[EpochRecord, ...]
    final_train_loss: float
    first_train_loss: float
    parameter_delta: float

    @property
    def loss_decreased(self) -> bool:
        return self.final_train_loss < self.first_train_loss


def _snapshot(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameter_delta(module: nn.Module, before: list[torch.Tensor]) -> float:
    total = 0.0
    for parameter, previous in zip(module.parameters(), before, strict=True):
        total += float(torch.sum((parameter.detach() - previous) ** 2))
    return float(np.sqrt(total))


def fit_supervised(
    module: nn.Module,
    features: np.ndarray,
    target: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: TrainConfig,
    shuffle: bool = True,
) -> TrainingReport:
    if features.shape[0] != target.shape[0]:
        raise ValueError("one target per example is required")
    if features.shape[0] == 0:
        raise ValueError("the training block is empty")
    set_seed(config.seed)
    device = torch.device(config.device)
    module.to(device)
    tensor_x = torch.as_tensor(features, dtype=torch.float32)
    tensor_y = torch.as_tensor(target, dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.seed)
    optimiser = build_optimiser(
        decay_groups(module, config.weight_decay), name="adamw", lr=config.lr
    )
    steps_per_epoch = max(1, int(np.ceil(features.shape[0] / config.batch_size)))
    total_steps = steps_per_epoch * config.epochs
    scheduler = cosine_with_warmup(
        optimiser, int(total_steps * config.warmup_fraction), total_steps
    )
    scaler = GradScaler(device.type, enabled=config.amp and device.type == "cuda")
    before = _snapshot(module)
    history: list[EpochRecord] = []
    for epoch in range(config.epochs):
        module.train()
        order = (
            torch.randperm(features.shape[0], generator=generator)
            if shuffle
            else torch.arange(features.shape[0])
        )
        running = 0.0
        seen = 0
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            batch_x = tensor_x[index].to(device)
            batch_y = tensor_y[index].to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                prediction = module(batch_x)
                loss = loss_fn(prediction, batch_y)
            torch.autograd.backward(scaler.scale(loss))
            if config.grad_clip > 0.0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(module.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += float(loss.detach()) * int(index.numel())
            seen += int(index.numel())
        record = EpochRecord(
            epoch=epoch + 1,
            train_loss=running / max(seen, 1),
            learning_rate=float(optimiser.param_groups[0]["lr"]),
        )
        history.append(record)
        if config.log_every > 0 and (epoch + 1) % config.log_every == 0:
            LOGGER.info("epoch %d loss %.6f", record.epoch, record.train_loss)
    return TrainingReport(
        history=tuple(history),
        final_train_loss=history[-1].train_loss,
        first_train_loss=history[0].train_loss,
        parameter_delta=_parameter_delta(module, before),
    )
