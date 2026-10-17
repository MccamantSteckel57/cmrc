"""Return-layer baselines: autoregression and a compact transformer encoder.

Ref: Sec. 4.2 (return component, AR and Transformer learners), Table 1 Panel B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from coupled_monotone_risk.models.base import scalar_head, transformer_stack, validate_sequence
from coupled_monotone_risk.utils.numeric import FloatArray


@dataclass
class AutoregressiveForecaster:
    lags: int = 5
    name: str = "ar"

    def __post_init__(self) -> None:
        self._coefficients: FloatArray | None = None
        self._intercept: float = 0.0

    def fit(self, returns: FloatArray) -> None:
        values = np.asarray(returns, dtype=np.float64)
        if values.size < self.lags + 10:
            raise ValueError("too few returns for the requested lag order")
        rows = values.size - self.lags
        design = np.column_stack(
            [np.ones(rows)] + [values[self.lags - lag : values.size - lag] for lag in range(1, self.lags + 1)]
        )
        response = values[self.lags :]
        solution, *_ = np.linalg.lstsq(design, response, rcond=None)
        self._intercept = float(solution[0])
        self._coefficients = np.asarray(solution[1:], dtype=np.float64)

    def filter(self, returns: FloatArray) -> FloatArray:
        if self._coefficients is None:
            raise RuntimeError("fit must be called before filter")
        values = np.asarray(returns, dtype=np.float64)
        forecast = np.full(values.size, np.nan, dtype=np.float64)
        for index in range(self.lags, values.size):
            history = values[index - self.lags : index][::-1]
            forecast[index] = self._intercept + float(np.dot(self._coefficients, history))
        return forecast


class ReturnTransformer(nn.Module):
    def __init__(
        self,
        context: int = 32,
        d_model: int = 32,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context = context
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, context, d_model))
        self.encoder = transformer_stack(d_model, heads, feedforward, dropout, layers)
        self.head = scalar_head(d_model)

    def forward(self, x: Tensor) -> Tensor:
        length = validate_sequence(x, self.context)
        hidden = self.input_projection(x.mean(dim=-1, keepdim=True))
        hidden = self.encoder(hidden + self.position[:, :length])
        output: Tensor = self.head(hidden[:, -1, :])
        return output.squeeze(-1)
