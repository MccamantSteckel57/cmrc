"""Selective state-space block with input-dependent decay and gating.

Ref: Sec. 4.2 (Mamba-2 backbone), Table 1 Panel A.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from coupled_monotone_risk.models.base import scalar_head, validate_sequence


class SelectiveStateSpaceBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand: int = 2,
        kernel: int = 4,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.inner = expand * d_model
        self.d_state = d_state
        self.in_projection = nn.Linear(d_model, 2 * self.inner)
        self.conv = nn.Conv1d(
            self.inner, self.inner, kernel, groups=self.inner, padding=kernel - 1
        )
        self.dt_projection = nn.Linear(self.inner, self.inner)
        self.b_projection = nn.Linear(self.inner, d_state)
        self.c_projection = nn.Linear(self.inner, d_state)
        self.a_log = nn.Parameter(
            torch.log(torch.linspace(1.0, float(d_state), d_state)).repeat(self.inner, 1)
        )
        self.d_skip = nn.Parameter(torch.ones(self.inner))
        self.out_projection = nn.Linear(self.inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        with torch.no_grad():
            self.dt_projection.bias.uniform_(math.log(dt_min), math.log(dt_max))
            bound = 1.0 / math.sqrt(self.inner)
            self.dt_projection.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("expected input of shape (batch, length, width)")
        batch, length, _ = x.shape
        value, gate = self.in_projection(x).chunk(2, dim=-1)
        convolved = self.conv(value.transpose(1, 2))[:, :, :length].transpose(1, 2)
        value = F.silu(convolved)
        delta = F.softplus(self.dt_projection(value))
        b_term = self.b_projection(value)
        c_term = self.c_projection(value)
        decay = -torch.exp(self.a_log)
        state = torch.zeros(batch, self.inner, self.d_state, dtype=x.dtype, device=x.device)
        collected: list[Tensor] = []
        for step in range(length):
            step_delta = delta[:, step, :].unsqueeze(-1)
            state = state * torch.exp(step_delta * decay) + step_delta * (
                b_term[:, step, :].unsqueeze(1) * value[:, step, :].unsqueeze(-1)
            )
            readout = (state * c_term[:, step, :].unsqueeze(1)).sum(dim=-1)
            collected.append(readout + self.d_skip * value[:, step, :])
        mixed = torch.stack(collected, dim=1)
        output: Tensor = self.norm(x + self.out_projection(mixed * F.silu(gate)))
        return output


class MambaForecaster(nn.Module):
    def __init__(
        self,
        context: int = 64,
        d_model: int = 64,
        d_state: int = 16,
        layers: int = 2,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context = context
        self.input_projection = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, context, d_model))
        self.blocks = nn.ModuleList(
            [SelectiveStateSpaceBlock(d_model=d_model, d_state=d_state, expand=expand) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = scalar_head(d_model)

    def forward(self, x: Tensor) -> Tensor:
        length = validate_sequence(x, self.context)
        hidden = self.input_projection(x.mean(dim=-1, keepdim=True))
        hidden = hidden + self.position[:, :length]
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.dropout(hidden[:, -1, :])
        projected: Tensor = self.head(hidden)
        return projected.squeeze(-1)
