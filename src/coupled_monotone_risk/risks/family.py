"""Nested prediction families indexed by one scalar per task (Sec. 3.1-3.2).

The interval family returns a symmetric interval around a point forecast, the set
family returns a label set from a scorer. Both expose a risk and a width that are
evaluated at a scalar level ``lam`` in [0, 1]; both are non-increasing and
right-continuous in ``lam`` by construction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from coupled_monotone_risk.utils.numeric import FloatArray, as_float_array

Mask = npt.NDArray[np.bool_]


@runtime_checkable
class NestedFamily(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def scores(self) -> FloatArray: ...

    def threshold(self, level: float) -> float: ...

    def risk(self, level: float, mask: Mask | None = None) -> float: ...

    def width(self, level: float, mask: Mask | None = None) -> float: ...

    def subset(self, mask: Mask) -> NestedFamily: ...


class IntervalFamily:
    """Interval miscoverage family over absolute residuals.

    Ref: Sec. 3.1 (interval task risk), Assumption 1.
    """

    def __init__(self, scores: npt.ArrayLike, width_scale: float) -> None:
        self._scores = as_float_array(scores)
        if self._scores.size < 2:
            raise ValueError("an interval family needs at least two residuals")
        if np.any(self._scores < 0.0):
            raise ValueError("absolute residuals must be non-negative")
        if width_scale <= 0.0:
            raise ValueError("width scale must be strictly positive")
        self._width_scale = float(width_scale)
        self._ordered = np.sort(self._scores)

    @property
    def size(self) -> int:
        return int(self._scores.size)

    @property
    def scores(self) -> FloatArray:
        return self._scores

    def threshold(self, level: float) -> float:
        clipped = float(np.clip(level, 0.0, 1.0))
        return float(np.quantile(self._ordered, clipped, method="linear"))

    def risk(self, level: float, mask: Mask | None = None) -> float:
        threshold = self.threshold(level)
        scores = self._scores if mask is None else self._scores[mask]
        if scores.size == 0:
            raise ValueError("empty mask leaves no residual to score")
        return float(np.mean(scores > threshold))

    def width(self, level: float, mask: Mask | None = None) -> float:
        return 2.0 * self.threshold(level) * self._width_scale

    def subset(self, mask: Mask) -> NestedFamily:
        if mask.size != self._scores.size:
            raise ValueError("the mask must cover every residual")
        return IntervalFamily(self._scores[mask], self._width_scale)


class SetFamily:
    """Set-misclassification family over a scorer's predicted probabilities.

    Ref: Sec. 3.1 (set task risk), Sec. 4.1 (credit set layer).
    """

    def __init__(self, probabilities: npt.ArrayLike, targets: npt.ArrayLike) -> None:
        matrix = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.int64)
        if matrix.ndim != 2:
            raise ValueError("probabilities must be a two-dimensional array")
        if matrix.shape[0] != labels.size:
            raise ValueError("probability rows must match the label count")
        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise ValueError("probabilities must lie in [0, 1]")
        if np.any(labels < 0) or np.any(labels >= matrix.shape[1]):
            raise ValueError("labels fall outside the score columns")
        self._probabilities = matrix
        self._targets = labels
        self._scores = 1.0 - matrix[np.arange(labels.size), labels]
        self._ordered = np.sort(self._scores)

    @property
    def size(self) -> int:
        return int(self._targets.size)

    @property
    def scores(self) -> FloatArray:
        return self._scores

    def threshold(self, level: float) -> float:
        clipped = float(np.clip(level, 0.0, 1.0))
        return float(np.quantile(self._ordered, clipped, method="linear"))

    def risk(self, level: float, mask: Mask | None = None) -> float:
        threshold = self.threshold(level)
        scores = self._scores if mask is None else self._scores[mask]
        if scores.size == 0:
            raise ValueError("empty mask leaves no label to score")
        return float(np.mean(scores > threshold))

    def width(self, level: float, mask: Mask | None = None) -> float:
        threshold = self.threshold(level)
        probability_rows = self._probabilities if mask is None else self._probabilities[mask]
        if probability_rows.shape[0] == 0:
            raise ValueError("empty mask leaves no set to size")
        included = probability_rows >= (1.0 - threshold)
        sizes = included.sum(axis=1)
        return float(np.mean(np.maximum(sizes, 1)))

    def subset(self, mask: Mask) -> NestedFamily:
        if mask.size != self._targets.size:
            raise ValueError("the mask must cover every observation")
        return SetFamily(self._probabilities[mask], self._targets[mask])

    def predicted_sets(self, level: float) -> npt.NDArray[np.bool_]:
        threshold = self.threshold(level)
        return np.asarray(self._probabilities >= (1.0 - threshold), dtype=np.bool_)


def family_from_predictions(
    kind: str,
    point: npt.ArrayLike,
    outcome: npt.ArrayLike,
    scale: npt.ArrayLike | None = None,
) -> NestedFamily:
    if kind == "interval":
        residual = np.abs(np.asarray(outcome, dtype=np.float64) - np.asarray(point, np.float64))
        if scale is None:
            raise ValueError("interval families need a width scale")
        return IntervalFamily(residual, float(np.mean(np.asarray(scale, dtype=np.float64))))
    if kind == "set":
        return SetFamily(point, outcome)
    raise ValueError(f"unknown task kind {kind!r}")
