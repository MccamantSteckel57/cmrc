"""Seed handling shared by every stochastic component."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> int:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def derive_seed(base: int, stream: int) -> int:
    """Independent deterministic seed per evaluation stream."""
    generator = np.random.default_rng([base, stream])
    return int(generator.integers(0, 2**31 - 1))
