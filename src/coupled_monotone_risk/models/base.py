"""Model interfaces used by the evaluation runner."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from torch import Tensor, nn

FloatArray = npt.NDArray[np.float64]


def validate_sequence(x: Tensor, context: int) -> int:
    if x.dim() != 3:
        raise ValueError("expected input of shape (batch, context, channels)")
    length = int(x.shape[1])
    if length != context:
        raise ValueError("input length does not match the configured context")
    return length


def transformer_stack(
    d_model: int, heads: int, feedforward: int, dropout: float, layers: int
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


def scalar_head(d_model: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
