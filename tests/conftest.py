"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from coupled_monotone_risk.risks.family import IntervalFamily, SetFamily


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture
def interval_pair(rng: np.random.Generator) -> tuple[IntervalFamily, IntervalFamily]:
    return (
        IntervalFamily(np.abs(rng.normal(0.0, 1.0, 600)), 1.0),
        IntervalFamily(np.abs(rng.normal(0.0, 2.0, 450)), 2.0),
    )


@pytest.fixture
def mixed_pair(rng: np.random.Generator) -> tuple[IntervalFamily, SetFamily]:
    probabilities = rng.dirichlet([1.0, 1.0], size=500)
    labels = rng.integers(0, 2, size=500)
    return IntervalFamily(np.abs(rng.normal(size=500)), 1.0), SetFamily(probabilities, labels)

