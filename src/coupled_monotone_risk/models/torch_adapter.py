"""Adapters that train and query a torch sequence module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from coupled_monotone_risk.training.ckpt import CheckpointMeta, load_checkpoint, save_checkpoint
from coupled_monotone_risk.training.trainer import TrainConfig, TrainingReport, fit_supervised
from coupled_monotone_risk.utils.numeric import FloatArray


@dataclass
class TorchRegressor:
    module: nn.Module
    config: TrainConfig
    name: str = "torch_regressor"

    def __post_init__(self) -> None:
        self._report: TrainingReport | None = None

    @property
    def report(self) -> TrainingReport:
        if self._report is None:
            raise RuntimeError("the regressor has not been fitted")
        return self._report

    def fit(self, windows: FloatArray, target: FloatArray) -> TrainingReport:
        report = fit_supervised(
            self.module,
            np.asarray(windows, dtype=np.float32),
            np.asarray(target, dtype=np.float32),
            nn.functional.mse_loss,
            self.config,
        )
        self._report = report
        return report

    def predict(self, windows: FloatArray) -> FloatArray:
        self.module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(windows, dtype=np.float32))
            output = self.module(tensor)
        return np.asarray(output.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    def save(self, path: Path, epoch: int) -> Path:
        return save_checkpoint(
            path,
            self.module,
            CheckpointMeta(epoch=epoch, seed=self.config.seed, extra={"model": self.name}),
        )

    def load(self, path: Path) -> CheckpointMeta:
        return load_checkpoint(path, self.module)
