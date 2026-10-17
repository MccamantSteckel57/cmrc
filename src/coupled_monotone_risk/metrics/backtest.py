"""Supervisory backtesting statistics (Table 4.6, Sec. 5.4).

Kupiec proportion-of-failures and Christoffersen independence, both on the
sequence of interval failures inside one regime window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from coupled_monotone_risk.metrics.interval import wald_interval


@dataclass(frozen=True)
class TestOutcome:
    statistic: float
    p_value: float
    degrees: int

    @property
    def rejects(self) -> bool:
        return self.p_value < 0.05


def kupiec_pof(failures: int, observations: int, nominal: float) -> TestOutcome:
    if observations <= 0:
        raise ValueError("observations must be positive")
    if failures < 0 or failures > observations:
        raise ValueError("failures must lie inside [0, observations]")
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal failure rate must lie inside (0, 1)")
    observed = failures / observations
    if failures == 0:
        statistic = -2.0 * observations * math.log(1.0 - nominal)
    elif failures == observations:
        statistic = -2.0 * observations * math.log(nominal)
    else:
        statistic = -2.0 * (
            (observations - failures) * math.log(1.0 - nominal)
            + failures * math.log(nominal)
            - (observations - failures) * math.log(1.0 - observed)
            - failures * math.log(observed)
        )
    return TestOutcome(float(max(statistic, 0.0)), float(stats.chi2.sf(statistic, 1)), 1)


def christoffersen_independence(hits: npt.ArrayLike) -> TestOutcome:
    values = np.asarray(hits, dtype=np.int64)
    if values.size < 10:
        raise ValueError("independence testing needs at least ten observations")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("transition counts need a binary failure indicator")
    previous, current = values[:-1], values[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    row_zero, row_one = n00 + n01, n10 + n11
    if row_zero == 0 or row_one == 0:
        return TestOutcome(0.0, 1.0, 1)
    probability = (n01 + n11) / (row_zero + row_one)
    probability_zero = n01 / row_zero
    probability_one = n11 / row_one
    log_likelihood_null = (n00 + n10) * math.log(1.0 - probability) if probability < 1.0 else 0.0
    log_likelihood_null += (n01 + n11) * math.log(probability) if probability > 0.0 else 0.0
    log_likelihood_alt = 0.0
    for count, ratio in ((n00, 1.0 - probability_zero), (n01, probability_zero)):
        if count > 0 and ratio > 0.0:
            log_likelihood_alt += count * math.log(ratio)
    for count, ratio in ((n10, 1.0 - probability_one), (n11, probability_one)):
        if count > 0 and ratio > 0.0:
            log_likelihood_alt += count * math.log(ratio)
    statistic = float(max(-2.0 * (log_likelihood_null - log_likelihood_alt), 0.0))
    return TestOutcome(statistic, float(stats.chi2.sf(statistic, 1)), 1)


def christoffersen_conditional(hits: npt.ArrayLike) -> TestOutcome:
    values = np.asarray(hits, dtype=np.int64)
    failure_rate = float(np.mean(values))
    pof = kupiec_pof(int(values.sum()), values.size, max(failure_rate, 1e-6))
    independence = christoffersen_independence(values)
    statistic = pof.statistic + independence.statistic
    return TestOutcome(float(statistic), float(stats.chi2.sf(statistic, 2)), 2)


def binomial_interval(hits: npt.ArrayLike) -> tuple[float, float]:
    return wald_interval(np.asarray(hits, dtype=np.float64))


def failure_sequence(lower: npt.ArrayLike, upper: npt.ArrayLike, actual: npt.ArrayLike) -> npt.NDArray[np.int64]:
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    return np.asarray(((observed < low) | (observed > high)).astype(np.int64), dtype=np.int64)
