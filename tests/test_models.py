import numpy as np
import pytest
import torch

from coupled_monotone_risk.data.features import har_design, har_log_design
from coupled_monotone_risk.models.arfima import (
    ArfimaForecaster,
    fractional_weights,
    gph_estimate,
)
from coupled_monotone_risk.models.char import CharForecaster
from coupled_monotone_risk.models.credit import CreditModel
from coupled_monotone_risk.models.har import HarForecaster
from coupled_monotone_risk.models.mamba2 import MambaForecaster
from coupled_monotone_risk.models.patchtst import PatchTST
from coupled_monotone_risk.models.realized_garch import RealizedGarchForecaster
from coupled_monotone_risk.models.recurrent import AutoregressiveForecaster, ReturnTransformer
from coupled_monotone_risk.models.torch_adapter import TorchRegressor
from coupled_monotone_risk.training.trainer import TrainConfig


def _variance_series(size: int = 400, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = np.abs(rng.normal(size=size))
    values = np.empty(size, dtype=np.float64)
    values[0] = 1.0
    for index in range(1, size):
        values[index] = 0.05 + 0.85 * values[index - 1] + 0.1 * shocks[index]
    return np.maximum(values, 1e-6)


def test_har_recovers_fitted_values() -> None:
    variance = _variance_series()
    design, target = har_design(variance)
    mask = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    model = HarForecaster()
    model.fit(design[mask], target[mask])
    prediction = model.predict(design[mask])
    assert prediction.shape == target[mask].shape
    assert np.all(prediction > 0.0)
    assert model.coefficients.size == 4


def test_har_log_outputs_positive_variance() -> None:
    variance = _variance_series()
    design, target = har_log_design(variance)
    mask = np.all(np.isfinite(design), axis=1) & np.isfinite(target)
    model = HarForecaster(name="har_log", log_target=True)
    model.fit(design[mask], target[mask])
    assert np.all(model.predict(design[mask]) > 0.0)


def test_har_rejects_too_few_rows() -> None:
    model = HarForecaster()
    with pytest.raises(ValueError):
        model.fit(np.ones((3, 4)), np.ones(3))


def test_char_filters_a_persistent_series() -> None:
    variance = _variance_series(300)
    model = CharForecaster()
    model.fit(variance[:200])
    forecast = model.filter(variance)
    assert forecast.shape == variance.shape
    assert np.all(np.isfinite(forecast))
    assert np.all(forecast > 0.0)
    assert model.parameters.size == 5


def test_realized_garch_runs_on_aligned_inputs() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, 220)
    variance = np.abs(rng.normal(1e-4, 2e-5, 220)) + 1e-8
    model = RealizedGarchForecaster()
    model.fit(returns, variance)
    filtered = model.filter(returns, variance)
    assert filtered.shape == returns.shape
    assert np.all(filtered > 0.0)


def test_fractional_weights_sum_positive() -> None:
    weights = fractional_weights(0.3, 50)
    assert weights.size == 50
    assert weights[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(weights))


def test_gph_finds_persistence() -> None:
    rng = np.random.default_rng(5)
    values = np.cumsum(rng.normal(size=500)) * 0.01 + rng.normal(size=500)
    estimate = gph_estimate(values)
    assert -0.45 <= estimate <= 0.45


def test_arfima_forecasts_on_the_log_scale() -> None:
    variance = _variance_series(260)
    model = ArfimaForecaster(truncation=40)
    model.fit(variance[:180])
    forecast = model.filter(variance)
    finite = forecast[np.isfinite(forecast)]
    assert finite.size > 50
    assert np.all(finite > 0.0)


def test_autoregression_recovers_a_known_process() -> None:
    rng = np.random.default_rng(6)
    values = np.zeros(300)
    for index in range(1, values.size):
        values[index] = 0.6 * values[index - 1] + rng.normal(0.0, 0.1)
    model = AutoregressiveForecaster(lags=1)
    model.fit(values)
    forecast = model.filter(values)
    assert np.isfinite(forecast[-1])
    assert abs(forecast[-1] - 0.6 * values[-2]) < 0.1


def test_patchtst_maps_a_batch_to_scalars() -> None:
    module = PatchTST(context=32, patch_length=8, stride=8, d_model=16, heads=2, layers=1)
    output = module(torch.randn(3, 32, 1))
    assert tuple(output.shape) == (3,)


def test_patchtst_rejects_mismatched_context() -> None:
    module = PatchTST(context=32, patch_length=8, stride=8, d_model=16, heads=2, layers=1)
    with pytest.raises(ValueError):
        module(torch.randn(2, 16, 1))


def test_mamba_outputs_one_scalar_per_example() -> None:
    module = MambaForecaster(context=16, d_model=8, d_state=4, layers=1)
    assert tuple(module(torch.randn(2, 16, 1)).shape) == (2,)


def test_transformer_outputs_one_scalar_per_example() -> None:
    module = ReturnTransformer(context=16, d_model=8, heads=2, layers=1)
    assert tuple(module(torch.randn(2, 16, 1)).shape) == (2,)


def test_torch_regressor_overfits_a_small_batch() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(16, 16, 1))
    target = windows[:, -1, 0] * 2.0
    module = MambaForecaster(context=16, d_model=8, d_state=4, layers=1)
    model = TorchRegressor(module=module, config=TrainConfig(epochs=40, batch_size=16, lr=5e-3, seed=7, log_every=0))
    report = model.fit(windows, target)
    assert report.final_train_loss < 0.5 * report.first_train_loss
    assert model.predict(windows).shape == (16,)


def test_credit_model_probabilities_are_simplex() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(64, 5))
    labels = (features[:, 0] > 0).astype(np.float64)
    model = CreditModel(width=5, config=TrainConfig(epochs=8, batch_size=32, lr=5e-3, seed=8, log_every=0), hidden=16)
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (64, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
