"""Tests for point-in-time-safe feature engineering."""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from quant_lab.ml.features import compute_features, build_training_set
from quant_lab.types import Bar


def _make_bars(
    symbol: str,
    n: int = 520,
    start_price: float = 100.0,
    trend: float = 0.0003,
    vol: float = 0.01,
    volume: int = 2_000_000,
    seed: int = 42,
) -> list[Bar]:
    """Generate `n` daily bars with a drift+noise price process."""
    import random
    rng = random.Random(seed)
    start = date(2022, 1, 3)
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


def _make_histories(n_symbols: int = 5, n_days: int = 520) -> dict[str, list[Bar]]:
    histories: dict[str, list[Bar]] = {}
    histories["SPY"] = _make_bars("SPY", n=n_days, seed=0)
    for i in range(n_symbols):
        sym = f"TICK{i}"
        histories[sym] = _make_bars(sym, n=n_days, seed=i + 1)
    return histories


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_features_returns_dataframe():
    histories = _make_histories(n_symbols=5, n_days=520)
    symbols = [s for s in histories if s != "SPY"]
    as_of = date(2022, 1, 3) + timedelta(days=400)

    X, y = compute_features(histories, symbols, as_of)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) > 0
    assert len(X) == len(y)


def test_compute_features_expected_columns():
    histories = _make_histories(n_symbols=3, n_days=520)
    symbols = [s for s in histories if s != "SPY"]
    as_of = date(2022, 1, 3) + timedelta(days=400)

    X, _ = compute_features(histories, symbols, as_of)

    expected_cols = [
        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "ret_252d",
        "vol_20d", "vol_60d", "vol_of_vol",
        "zscore_20", "zscore_60", "zscore_200",
        "dist_sma_20_vol", "dist_sma_60_vol",
        "rel_volume", "obv_change", "log_adv",
        "rsi_14", "macd_line", "macd_signal",
        "bb_zscore", "atr", "atr_pct",
        "rel_ret_5d", "rel_ret_20d",
    ]
    for col in expected_cols:
        assert col in X.columns, f"Missing expected feature column: {col}"


def test_compute_features_no_nan_in_core_features():
    """With sufficient history (520 days), core return/vol features should not be NaN."""
    histories = _make_histories(n_symbols=3, n_days=520)
    symbols = [s for s in histories if s != "SPY"]
    as_of = date(2022, 1, 3) + timedelta(days=400)

    X, _ = compute_features(histories, symbols, as_of)

    # Core features that should always be finite with 400+ bars
    core_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "rsi_14", "bb_zscore"]
    for col in core_cols:
        assert col in X.columns
        nan_count = X[col].isna().sum()
        assert nan_count == 0, f"Unexpected NaN in {col}: {nan_count} NaN values"


def test_compute_features_insufficient_history_returns_empty():
    """Symbols with fewer than 252 bars should be excluded."""
    histories = {
        "SPY": _make_bars("SPY", n=520, seed=0),
        "SHORT": _make_bars("SHORT", n=100, seed=1),  # too short
    }
    as_of = date(2022, 1, 3) + timedelta(days=80)

    X, y = compute_features(histories, ["SHORT"], as_of)
    assert X.empty
    assert y.empty


def test_compute_features_point_in_time_strict():
    """Features computed on truncated history must match features on full history at same as_of."""
    histories_full = _make_histories(n_symbols=2, n_days=520)
    symbols = ["TICK0", "TICK1"]
    as_of = date(2022, 1, 3) + timedelta(days=400)

    # Full history
    X_full, _ = compute_features(histories_full, symbols, as_of)

    # Truncated history: cut off everything after as_of (simulating "we don't have future data")
    histories_trunc = {}
    for sym, bars in histories_full.items():
        histories_trunc[sym] = [b for b in bars if b.date <= as_of]

    X_trunc, _ = compute_features(histories_trunc, symbols, as_of)

    # Both should produce the same feature values
    assert set(X_full.index) == set(X_trunc.index)
    for sym in X_full.index:
        for col in X_full.columns:
            v_full = X_full.loc[sym, col]
            v_trunc = X_trunc.loc[sym, col]
            if math.isnan(v_full) and math.isnan(v_trunc):
                continue
            assert abs(v_full - v_trunc) < 1e-9, (
                f"Point-in-time violation: {sym}/{col} full={v_full} trunc={v_trunc}"
            )


def test_label_uses_future_data():
    """y should be NaN when there's no future data beyond as_of."""
    histories = _make_histories(n_symbols=2, n_days=520)
    symbols = ["TICK0", "TICK1"]
    # Set as_of to the very last bar date (no future data)
    last_date = max(b.date for bars in histories.values() for b in bars)

    _, y = compute_features(histories, symbols, as_of=last_date, horizon=5)
    # With as_of = last date, there are no 5 future bars → labels should all be NaN
    assert y.isna().all()


def test_build_training_set_shape():
    """build_training_set should return a non-empty X/y with valid labels."""
    histories = _make_histories(n_symbols=3, n_days=520)
    symbols = [s for s in histories if s != "SPY"]
    base = date(2022, 1, 3)
    train_start = base + timedelta(days=260)
    train_end = base + timedelta(days=400)

    X, y = build_training_set(
        histories=histories,
        target_symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        horizon=5,
        sample_every_days=5,
    )

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)
    # All labels should be finite (NaN rows dropped)
    assert y.notna().all()


def test_index_proxies_excluded_from_features():
    """SPY, QQQ, ^VIX should not appear in feature output."""
    histories = {
        "SPY": _make_bars("SPY", n=520, seed=0),
        "QQQ": _make_bars("QQQ", n=520, seed=1),
        "^VIX": _make_bars("^VIX", n=520, seed=2),
        "AAPL": _make_bars("AAPL", n=520, seed=3),
    }
    as_of = date(2022, 1, 3) + timedelta(days=400)

    X, _ = compute_features(histories, list(histories.keys()), as_of)

    assert "SPY" not in X.index
    assert "QQQ" not in X.index
    assert "^VIX" not in X.index
