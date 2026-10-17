"""Table assembly from executed arms; no quoted benchmark constants are stored."""

from __future__ import annotations

from collections.abc import Sequence

from coupled_monotone_risk.evaluation.protocol import ArmResult, RegimeCell


def main_rows(arms: Sequence[ArmResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in arms:
        rows.append(
            {
                "arm": arm.arm,
                "task": arm.task,
                "kind": arm.kind,
                "asset": arm.asset,
                "level": round(arm.level, 4),
                "coverage": round(arm.joint_coverage, 4),
                "width": round(arm.joint_width, 4),
                "worst_gap_pp": round(arm.worst_gap_pp, 3),
            }
        )
    return rows


def regime_rows(arms: Sequence[ArmResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in arms:
        for cell in arm.cells:
            rows.append(_cell_row(arm, cell))
    return rows


def _cell_row(arm: ArmResult, cell: RegimeCell) -> dict[str, object]:
    return {
        "task": arm.task,
        "arm": arm.arm,
        "regime": cell.regime,
        "n": cell.n,
        "coverage": round(cell.coverage, 4),
        "wald_low": round(cell.wald_low, 4),
        "wald_high": round(cell.wald_high, 4),
        "worst_gap_pp": round(cell.gap_pp, 3),
        "mean_width": round(cell.width, 4),
        "underpowered": cell.underpowered,
    }
