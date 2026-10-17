"""Credit scorer and its probed two-class output (Table 1, Panel D)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from coupled_monotone_risk.training.trainer import TrainConfig, TrainingReport, fit_supervised
from coupled_monotone_risk.utils.numeric import FloatArray


class CreditScorer(nn.Module):
    def __init__(self, width: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("the feature width must be positive")
        blocks: list[nn.Module] = []
        current = width
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(current, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden
        blocks.append(nn.Linear(current, 2))
        self.network = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.network(x)
        return output


@dataclass
class CreditModel:
    width: int
    config: TrainConfig
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        self._module = CreditScorer(self.width, self.hidden, self.layers, self.dropout)
        self._report: TrainingReport | None = None

    @property
    def module(self) -> CreditScorer:
        return self._module

    @property
    def report(self) -> TrainingReport:
        if self._report is None:
            raise RuntimeError("the scorer has not been fitted")
        return self._report

    def fit(self, features: FloatArray, labels: FloatArray) -> TrainingReport:
        weights = torch.as_tensor(
            np.asarray([1.0, float(np.mean(labels == 0) / max(np.mean(labels == 1), 1e-6))]),
            dtype=torch.float32,
        )

        def loss_fn(logits: Tensor, target: Tensor) -> Tensor:
            return F.cross_entropy(logits, target.long(), weight=weights)

        report = fit_supervised(
            self._module,
            np.asarray(features, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            loss_fn,
            self.config,
        )
        self._report = report
        return report

    def predict_proba(self, features: FloatArray) -> FloatArray:
        self._module.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(features, dtype=np.float32))
            logits = self._module(tensor)
            probability = torch.softmax(logits, dim=-1)
        return np.asarray(probability.numpy(), dtype=np.float64)

    def predict(self, features: FloatArray) -> FloatArray:
        return self.predict_proba(features)[:, 1]
