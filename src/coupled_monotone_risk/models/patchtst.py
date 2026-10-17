"""PatchTST encoder with channel-independent patching and RevIN.

Ref: Sec. 4.2 (patch-transformer backbone), Table 1 Panel A.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from coupled_monotone_risk.models.base import transformer_stack, validate_sequence


class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean = x.mean(dim=1, keepdim=True)
        deviation = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / deviation, mean, deviation

    @staticmethod
    def inverse(output: Tensor, mean: Tensor, deviation: Tensor) -> Tensor:
        return output * deviation + mean


class PatchTST(nn.Module):
    def __init__(
        self,
        context: int = 64,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 128,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context < patch_length:
            raise ValueError("the context must cover at least one patch")
        if (context - patch_length) % stride != 0:
            raise ValueError("context, patch length and stride must tile exactly")
        self.context = context
        self.patch_length = patch_length
        self.stride = stride
        self.patches = (context - patch_length) // stride + 1
        self.revin = RevIN()
        self.embed = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.patches, d_model))
        self.encoder = transformer_stack(d_model, heads, feedforward, dropout, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(self.patches * d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        length = validate_sequence(x, self.context)
        batch, _, channels = x.shape
        flattened = x.permute(0, 2, 1).reshape(batch * channels, length)
        normalised, mean, deviation = self.revin(flattened)
        tokens = normalised.unfold(1, self.patch_length, self.stride)
        hidden = self.embed(tokens) + self.position[:, : tokens.shape[1]]
        encoded = self.norm(self.encoder(hidden))
        projected: Tensor = self.head(encoded)
        restored = projected * deviation + mean
        flattened_output: Tensor = restored.reshape(batch, channels).mean(dim=1)
        return flattened_output
