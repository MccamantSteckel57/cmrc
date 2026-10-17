"""Fit one forecaster on a named panel and store a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coupled_monotone_risk.cli.common import add_common, emit, resolve_config
from coupled_monotone_risk.data.credit import load_german, load_taiwan
from coupled_monotone_risk.data.manifest import MissingDataError
from coupled_monotone_risk.evaluation.protocol import ProtocolConfig
from coupled_monotone_risk.models.credit import CreditModel
from coupled_monotone_risk.training.ckpt import CheckpointMeta, save_checkpoint
from coupled_monotone_risk.utils.config import lookup
from coupled_monotone_risk.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fit a credit scorer on one panel")
    add_common(parser)
    parser.add_argument("--panel", choices=("german", "taiwan"), default="taiwan")
    parser.add_argument("--checkpoint", type=Path, default=Path("artefacts/scorer.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = resolve_config(arguments)
    epochs = int(config.get("train", {}).get("epochs", arguments.epochs))
    batch = int(config.get("train", {}).get("batch_size", 256))
    learning_rate = float(config.get("train", {}).get("lr", 1e-3))
    seed = int(config.get("train", {}).get("seed", arguments.seed))
    protocol = ProtocolConfig(
        epochs=epochs, batch_size=batch, lr=learning_rate, seed=seed
    )
    loaders = {"german": load_german, "taiwan": load_taiwan}
    try:
        panel = loaders[arguments.panel]()
    except (MissingDataError, FileNotFoundError) as error:
        LOGGER.error("panel unavailable: %s", error)
        return 2
    model = CreditModel(
        width=int(panel.features.shape[1]),
        config=protocol.train_config(),
        hidden=int(lookup(config, "train.hidden", 64)),
    )
    report = model.fit(panel.features, panel.target.astype(np.float64))
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        arguments.checkpoint,
        model.module,
        CheckpointMeta(epoch=epochs, seed=seed, extra={"panel": panel.name}),
    )
    emit(
        {
            "panel": panel.name,
            "rows": panel.size,
            "features": int(panel.features.shape[1]),
            "prevalence": round(panel.prevalence, 5),
            "first_loss": round(report.first_train_loss, 6),
            "final_loss": round(report.final_train_loss, 6),
            "parameter_delta": round(report.parameter_delta, 6),
            "checkpoint": str(arguments.checkpoint),
        },
        arguments.output,
    )
    return 0 if report.loss_decreased else 1


if __name__ == "__main__":
    raise SystemExit(main())
