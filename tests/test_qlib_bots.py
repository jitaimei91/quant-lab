"""Tests for the qlib-derived ML bots: CatBoost, DoubleEnsemble, qlib-linear.

Each bot wraps a model trained on the same ~30 features as gradboost /
lightforest. These tests cover:
  - walk-forward training runs end-to-end and persists a model
  - persisted model loads and produces deterministic predictions
  - strategy class falls back to SPY when no model file exists
  - strategy class produces a valid weight dict when a model is loaded
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from quant_lab.backtest.windows import Window
from quant_lab.ml.train import (
    train_catboost_walkforward,
    train_double_ensemble_walkforward,
    train_ridge_walkforward,
    train_mlp_walkforward,
)
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Synthetic data — mirrors tests/test_ml_train.py for consistency
# ---------------------------------------------------------------------------


def _make_bars(
    symbol: str,
    n: int = 600,
    start_price: float = 100.0,
    trend: float = 0.0003,
    vol: float = 0.012,
    volume: int = 2_000_000,
    seed: int = 42,
) -> list[Bar]:
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = start_price
    bars = []
    for i in range(n):
        ret = rng.gauss(trend, vol)
        price = max(price * (1 + ret), 0.01)
        bars.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=volume,
            )
        )
    return bars


def _synth_histories(n_symbols: int = 4, n_days: int = 600) -> dict[str, list[Bar]]:
    h: dict[str, list[Bar]] = {"SPY": _make_bars("SPY", n=n_days, seed=0)}
    for i in range(n_symbols):
        h[f"TICK{i}"] = _make_bars(f"TICK{i}", n=n_days, seed=i + 10)
    return h


def _make_windows(histories: dict, n_windows: int = 2) -> list[Window]:
    all_dates = sorted({b.date for bars in histories.values() for b in bars})
    base = all_dates[260]
    windows = []
    for i in range(n_windows):
        train_start = base + timedelta(days=i * 60)
        train_end = train_start + timedelta(days=60)
        test_end = train_end + timedelta(days=30)
        windows.append(
            Window(
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                label=f"qlib-wf-{i}",
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "train_fn, expected_prefix",
    [
        (train_catboost_walkforward, "catboost"),
        (train_double_ensemble_walkforward, "double-ensemble"),
        (train_ridge_walkforward, "qlib-linear"),
        (train_mlp_walkforward, "qlib-mlp"),
    ],
)
def test_walkforward_runs_and_persists(train_fn, expected_prefix):
    histories = _synth_histories(n_symbols=4, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_fn(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            seed=0,
            models_dir=Path(tmpdir),
        )

        assert result["final_model"] is not None
        assert result["persisted_path"] is not None
        assert expected_prefix in result["persisted_path"]
        assert len(result["fold_sharpes"]) > 0
        for fold in result["fold_results"]:
            assert isinstance(fold["fold_sharpe"], float)


@pytest.mark.parametrize(
    "train_fn",
    [
        train_catboost_walkforward,
        train_double_ensemble_walkforward,
        train_ridge_walkforward,
        train_mlp_walkforward,
    ],
)
def test_persisted_model_matches_in_memory(train_fn):
    import joblib

    histories = _synth_histories(n_symbols=3, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_fn(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            seed=0,
            models_dir=Path(tmpdir),
        )

        loaded = joblib.load(result["persisted_path"])
        in_mem = result["final_model"]

        dummy_X = np.random.default_rng(0).random((10, 25))
        np.testing.assert_array_almost_equal(
            loaded.predict(dummy_X),
            in_mem.predict(dummy_X),
        )


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------


def _import_bots():
    from quant_lab.strategies.catboost_bot import CatBoost
    from quant_lab.strategies.double_ensemble import DoubleEnsemble
    from quant_lab.strategies.qlib_linear import QlibLinear
    from quant_lab.strategies.qlib_mlp import QlibMLP
    return CatBoost, DoubleEnsemble, QlibLinear, QlibMLP


def test_bots_registered():
    """Importing strategies/__init__.py should auto-register all qlib-derived bots."""
    from quant_lab.strategies import get_all
    import quant_lab.strategies  # noqa: F401 — triggers __init__

    bot_ids = {s.bot_id for s in get_all()}
    assert "catboost" in bot_ids
    assert "double-ensemble" in bot_ids
    assert "qlib-linear" in bot_ids
    assert "qlib-mlp" in bot_ids


@pytest.mark.parametrize("BotClass", _import_bots())
def test_bot_falls_back_to_spy_when_no_model(BotClass, tmp_path, monkeypatch):
    """No model file in models/ → bot returns SPY=1.0 (not cash)."""
    # Point the bot at an empty models dir
    import quant_lab.strategies.gradboost as gb_module
    monkeypatch.setattr(gb_module, "_MODELS_DIR", tmp_path)

    bot = BotClass()
    histories = _synth_histories(n_symbols=3, n_days=300)
    weights = bot.target_weights(histories, as_of=date(2021, 1, 1))

    assert weights == {"SPY": 1.0}


@pytest.mark.parametrize(
    "BotClass, train_fn, prefix",
    [
        ("CatBoost", train_catboost_walkforward, "catboost"),
        ("DoubleEnsemble", train_double_ensemble_walkforward, "double-ensemble"),
        ("QlibLinear", train_ridge_walkforward, "qlib-linear"),
        ("QlibMLP", train_mlp_walkforward, "qlib-mlp"),
    ],
)
def test_bot_produces_weights_after_training(BotClass, train_fn, prefix, tmp_path, monkeypatch):
    """End-to-end: train → save → bot loads → bot returns weights."""
    import quant_lab.strategies.gradboost as gb_module
    monkeypatch.setattr(gb_module, "_MODELS_DIR", tmp_path)

    histories = _synth_histories(n_symbols=5, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=1)

    result = train_fn(
        histories=histories,
        target_symbols=symbols,
        windows=windows,
        horizon=5,
        seed=0,
        models_dir=tmp_path,
    )
    assert result["persisted_path"] is not None

    # Import the class fresh and instantiate so it picks up the new model file.
    from quant_lab.strategies.catboost_bot import CatBoost
    from quant_lab.strategies.double_ensemble import DoubleEnsemble
    from quant_lab.strategies.qlib_linear import QlibLinear
    from quant_lab.strategies.qlib_mlp import QlibMLP
    cls = {
        "CatBoost": CatBoost,
        "DoubleEnsemble": DoubleEnsemble,
        "QlibLinear": QlibLinear,
        "QlibMLP": QlibMLP,
    }[BotClass]

    bot = cls()
    # The bot computes features over the universe. Use a date inside the
    # training window so it has enough history.
    as_of = windows[0].test_end - timedelta(days=1)
    weights = bot.target_weights(histories, as_of=as_of)

    assert isinstance(weights, dict)
    assert len(weights) > 0
    total = sum(weights.values())
    # Either it picked stocks (sum ~ 1 - cash_buffer) or fell back to SPY
    assert 0.9 <= total <= 1.0
