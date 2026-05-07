"""Tests for walk-forward training (XGBoost + LightGBM)."""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from quant_lab.backtest.windows import Window
from quant_lab.ml.train import train_xgboost_walkforward, train_lightgbm_walkforward
from quant_lab.types import Bar


def _make_bars(
    symbol: str,
    n: int = 520,
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
        bar_date = start + timedelta(days=i)
        bars.append(
            Bar(
                symbol=symbol,
                date=bar_date,
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=volume,
            )
        )
    return bars


def _synth_histories(n_symbols: int = 5, n_days: int = 600) -> dict[str, list[Bar]]:
    h: dict[str, list[Bar]] = {}
    h["SPY"] = _make_bars("SPY", n=n_days, seed=0)
    for i in range(n_symbols):
        h[f"TICK{i}"] = _make_bars(f"TICK{i}", n=n_days, seed=i + 10)
    return h


def _make_windows(histories: dict, n_windows: int = 2) -> list[Window]:
    """Create small walk-forward windows suitable for fast testing."""
    all_dates = sorted({b.date for bars in histories.values() for b in bars})
    base = all_dates[260]  # skip first ~260 bars (need 252 for features)
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
                label=f"test-wf-{i}",
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_xgboost_walkforward_runs_without_error():
    histories = _synth_histories(n_symbols=4, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_xgboost_walkforward(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            seed=0,
            models_dir=Path(tmpdir),
        )

    assert "fold_results" in result
    assert "fold_sharpes" in result
    assert "final_model" in result
    assert result["final_model"] is not None


def test_lightgbm_walkforward_runs_without_error():
    histories = _synth_histories(n_symbols=4, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_lightgbm_walkforward(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            seed=0,
            models_dir=Path(tmpdir),
        )

    assert result["final_model"] is not None
    assert len(result["fold_sharpes"]) > 0


def test_persisted_model_predicts_same_as_in_memory():
    """Saved model should produce identical predictions to the in-memory model."""
    import joblib
    import numpy as np

    histories = _synth_histories(n_symbols=3, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_xgboost_walkforward(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            seed=0,
            models_dir=Path(tmpdir),
        )

        assert result["persisted_path"] is not None
        loaded_model = joblib.load(result["persisted_path"])
        in_memory_model = result["final_model"]

        # Predict on a small dummy X
        dummy_X = np.random.default_rng(0).random((10, 25))
        preds_mem = in_memory_model.predict(dummy_X)
        preds_disk = loaded_model.predict(dummy_X)

        np.testing.assert_array_almost_equal(preds_mem, preds_disk)


def test_walkforward_produces_per_fold_metrics():
    histories = _synth_histories(n_symbols=4, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = train_xgboost_walkforward(
            histories=histories,
            target_symbols=symbols,
            windows=windows,
            horizon=5,
            models_dir=Path(tmpdir),
        )

    # Each fold result should have per-fold Sharpe
    for fold in result["fold_results"]:
        assert "fold_sharpe" in fold
        assert isinstance(fold["fold_sharpe"], float)
        assert "n_train_samples" in fold
        assert fold["n_train_samples"] > 0


def test_model_name_in_result():
    histories = _synth_histories(n_symbols=2, n_days=600)
    symbols = [s for s in histories if s != "SPY"]
    windows = _make_windows(histories, n_windows=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        xgb_result = train_xgboost_walkforward(
            histories=histories, target_symbols=symbols, windows=windows, models_dir=Path(tmpdir)
        )
        lgb_result = train_lightgbm_walkforward(
            histories=histories, target_symbols=symbols, windows=windows, models_dir=Path(tmpdir)
        )

    assert xgb_result["model_name"] == "gradboost"
    assert lgb_result["model_name"] == "lightforest"
