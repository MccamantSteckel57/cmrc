from pathlib import Path

import numpy as np
import pytest
import torch

from coupled_monotone_risk.models.mamba2 import MambaForecaster
from coupled_monotone_risk.training.ckpt import CheckpointMeta, load_checkpoint, save_checkpoint
from coupled_monotone_risk.training.optim import build_optimiser, decay_groups
from coupled_monotone_risk.training.sched import cosine_with_warmup, linear_warmup
from coupled_monotone_risk.training.trainer import TrainConfig, fit_supervised
from coupled_monotone_risk.utils.config import load_config, lookup
from coupled_monotone_risk.utils.seeding import derive_seed, set_seed

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _smoke_config() -> dict:
    return load_config([CONFIG_ROOT / "study" / "_smoke.yaml"])


def test_smoke_config_is_labelled_for_tests() -> None:
    config = _smoke_config()
    assert lookup(config, "study.name") == "smoke"
    assert lookup(config, "study.reporting") is False


def test_two_step_training_reduces_the_loss() -> None:
    config = _smoke_config()
    set_seed(int(lookup(config, "train.seed")))
    rng = np.random.default_rng(int(lookup(config, "train.seed")))
    windows = rng.normal(size=(24, 16, 1))
    target = windows[:, -1, 0]
    module = MambaForecaster(context=16, d_model=8, d_state=4, layers=1)
    train_config = TrainConfig(
        epochs=2,
        batch_size=8,
        lr=float(lookup(config, "train.lr")),
        seed=int(lookup(config, "train.seed")),
        log_every=0,
    )
    report = fit_supervised(module, windows, target, torch.nn.functional.mse_loss, train_config)
    assert len(report.history) == 2
    assert report.final_train_loss < report.first_train_loss
    assert report.parameter_delta > 0.0


def test_checkpoint_restores_weights_and_seed(tmp_path: Path) -> None:
    module = MambaForecaster(context=8, d_model=8, d_state=4, layers=1)
    path = tmp_path / "model.pt"
    save_checkpoint(path, module, CheckpointMeta(epoch=2, seed=19, extra={"stage": "smoke"}))
    reference = [parameter.detach().clone() for parameter in module.parameters()]
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
    meta = load_checkpoint(path, module)
    assert meta.seed == 19
    assert meta.epoch == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(reference, module.parameters(), strict=True)
    )


def test_optimiser_and_schedules() -> None:
    module = MambaForecaster(context=8, d_model=8, d_state=4, layers=1)
    optimiser = build_optimiser(decay_groups(module, 0.01), "adamw", lr=1e-3)
    scheduler = cosine_with_warmup(optimiser, 2, 10)
    warmup = linear_warmup(optimiser, 5)
    for _ in range(3):
        optimiser.step()
        scheduler.step()
        warmup.step()
    assert 0.0 < optimiser.param_groups[0]["lr"] <= 1e-3


def test_seed_derivation_is_reproducible() -> None:
    assert derive_seed(3, 7) == derive_seed(3, 7)
    assert derive_seed(3, 7) != derive_seed(3, 8)


def test_seed_rejects_negative() -> None:
    with pytest.raises(ValueError):
        set_seed(-1)
