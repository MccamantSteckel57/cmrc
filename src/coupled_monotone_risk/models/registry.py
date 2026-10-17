"""Name-based construction of the forecasters used by the protocol grid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from coupled_monotone_risk.models.arfima import ArfimaForecaster
from coupled_monotone_risk.models.char import CharForecaster
from coupled_monotone_risk.models.credit import CreditModel
from coupled_monotone_risk.models.har import HarForecaster
from coupled_monotone_risk.models.mamba2 import MambaForecaster
from coupled_monotone_risk.models.patchtst import PatchTST
from coupled_monotone_risk.models.realized_garch import RealizedGarchForecaster
from coupled_monotone_risk.models.recurrent import AutoregressiveForecaster, ReturnTransformer
from coupled_monotone_risk.models.torch_adapter import TorchRegressor
from coupled_monotone_risk.training.trainer import TrainConfig

Family = Literal["design", "recursive", "sequence", "credit"]


@dataclass(frozen=True)
class ModelEntry:
    name: str
    family: Family
    builder: str


MODELS: dict[str, ModelEntry] = {
    "har": ModelEntry("har", "design", "har"),
    "har_log": ModelEntry("har_log", "design", "har_log"),
    "char": ModelEntry("char", "recursive", "char"),
    "realized_garch": ModelEntry("realized_garch", "recursive", "realized_garch"),
    "arfima": ModelEntry("arfima", "recursive", "arfima"),
    "ar": ModelEntry("ar", "recursive", "ar"),
    "patchtst": ModelEntry("patchtst", "sequence", "patchtst"),
    "mamba2": ModelEntry("mamba2", "sequence", "mamba2"),
    "transformer": ModelEntry("transformer", "sequence", "transformer"),
    "credit_mlp": ModelEntry("credit_mlp", "credit", "credit_mlp"),
}


def _train_config(config: Mapping[str, Any]) -> TrainConfig:
    return TrainConfig(
        epochs=int(config.get("epochs", 60)),
        batch_size=int(config.get("batch_size", 64)),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
        warmup_fraction=float(config.get("warmup_fraction", 0.05)),
        grad_clip=float(config.get("grad_clip", 1.0)),
        seed=int(config.get("seed", 13)),
        device=str(config.get("device", "cpu")),
        amp=bool(config.get("amp", False)),
        log_every=int(config.get("log_every", 0)),
    )


def build(name: str, config: Mapping[str, Any] | None = None) -> Any:
    settings: Mapping[str, Any] = {} if config is None else config
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}")
    if name == "har":
        return HarForecaster(name="har", log_target=False)
    if name == "har_log":
        return HarForecaster(name="har_log", log_target=True)
    if name == "char":
        return CharForecaster()
    if name == "realized_garch":
        return RealizedGarchForecaster()
    if name == "arfima":
        return ArfimaForecaster(truncation=int(settings.get("truncation", 100)))
    if name == "ar":
        return AutoregressiveForecaster(lags=int(settings.get("lags", 5)))
    if name == "patchtst":
        patch_module = PatchTST(
            context=int(settings.get("context", 64)),
            patch_length=int(settings.get("patch_length", 16)),
            stride=int(settings.get("stride", 8)),
            d_model=int(settings.get("d_model", 64)),
            heads=int(settings.get("heads", 4)),
            layers=int(settings.get("layers", 2)),
            feedforward=int(settings.get("feedforward", 128)),
            dropout=float(settings.get("dropout", 0.1)),
            head_dropout=float(settings.get("head_dropout", 0.0)),
        )
        return TorchRegressor(module=patch_module, config=_train_config(settings), name="patchtst")
    if name == "mamba2":
        mamba_module = MambaForecaster(
            context=int(settings.get("context", 64)),
            d_model=int(settings.get("d_model", 64)),
            d_state=int(settings.get("d_state", 16)),
            layers=int(settings.get("layers", 2)),
            expand=int(settings.get("expand", 2)),
            dropout=float(settings.get("dropout", 0.0)),
        )
        return TorchRegressor(module=mamba_module, config=_train_config(settings), name="mamba2")
    if name == "transformer":
        return_module = ReturnTransformer(
            context=int(settings.get("context", 32)),
            d_model=int(settings.get("d_model", 32)),
            heads=int(settings.get("heads", 4)),
            layers=int(settings.get("layers", 2)),
            feedforward=int(settings.get("feedforward", 64)),
            dropout=float(settings.get("dropout", 0.1)),
        )
        return TorchRegressor(module=return_module, config=_train_config(settings), name="transformer")
    if name == "credit_mlp":
        width = int(settings["width"])
        return CreditModel(
            width=width,
            config=_train_config(settings),
            hidden=int(settings.get("hidden", 64)),
            layers=int(settings.get("layers", 2)),
            dropout=float(settings.get("dropout", 0.1)),
        )
    raise KeyError(f"model {name!r} has no builder")


def names() -> tuple[str, ...]:
    return tuple(MODELS)
