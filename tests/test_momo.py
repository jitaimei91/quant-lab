"""Tests for Momo — cross-sectional 6-month momentum strategy."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.strategies.momo import Momo
from quant_lab.types import Bar


def _make_bars(
    symbol: str,
    n: int = 300,
    start_price: float = 100.0,
    end_price: float = 110.0,
    volume: int = 1_000_000,
) -> list[Bar]:
    """Generate `n` daily bars with price linearly interpolated from start to end."""
    start = date(2024, 1, 2)
    bars = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        price = start_price + frac * (end_price - start_price)
        bars.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=volume,
            )
        )
    return bars


def test_strong_momentum_ticker_selected():
    """Ticker with clear upward momentum should be selected; flat ticker should not."""
    as_of = date(2024, 1, 2) + timedelta(days=299)
    # MOVER: price doubles → strong positive momentum
    # FLAT: price stays at 100 → near-zero momentum
    histories = {
        "MOVER": _make_bars("MOVER", n=300, start_price=50.0, end_price=100.0, volume=2_000_000),
        "FLAT": _make_bars("FLAT", n=300, start_price=100.0, end_price=100.1, volume=2_000_000),
        "SPY": _make_bars("SPY", n=300, start_price=400.0, end_price=410.0, volume=50_000_000),
    }
    strategy = Momo()
    weights = strategy.target_weights(histories, as_of)
    assert "MOVER" in weights
    assert "SPY" not in weights


def test_insufficient_history_returns_empty():
    """When all tickers have fewer than min_history_days bars, return {}."""
    as_of = date(2024, 6, 1)
    histories = {
        "AAPL": _make_bars("AAPL", n=100, volume=2_000_000),
    }
    strategy = Momo()
    weights = strategy.target_weights(histories, as_of)
    assert weights == {}


def test_weights_sum_within_cash_buffer():
    """Total weight must be <= 0.95 (strategy.cash_buffer = 0.05)."""
    as_of = date(2024, 1, 2) + timedelta(days=299)
    histories = {
        f"TICK{i}": _make_bars(
            f"TICK{i}",
            n=300,
            start_price=100.0,
            end_price=100.0 + i * 5,
            volume=2_000_000,
        )
        for i in range(1, 21)
    }
    strategy = Momo()
    weights = strategy.target_weights(histories, as_of)
    assert sum(weights.values()) <= 0.95 + 1e-9


def test_index_proxies_excluded():
    """SPY, QQQ, ^VIX must never appear in output weights."""
    as_of = date(2024, 1, 2) + timedelta(days=299)
    histories = {
        "SPY": _make_bars("SPY", n=300, start_price=100.0, end_price=200.0, volume=50_000_000),
        "QQQ": _make_bars("QQQ", n=300, start_price=100.0, end_price=300.0, volume=50_000_000),
        "^VIX": _make_bars("^VIX", n=300, start_price=15.0, end_price=40.0, volume=1_000_000),
        "AAPL": _make_bars("AAPL", n=300, start_price=100.0, end_price=120.0, volume=2_000_000),
    }
    strategy = Momo()
    weights = strategy.target_weights(histories, as_of)
    assert "SPY" not in weights
    assert "QQQ" not in weights
    assert "^VIX" not in weights


def test_adv_floor_filters_low_volume():
    """Tickers below the ADV floor ($5M) should be excluded."""
    as_of = date(2024, 1, 2) + timedelta(days=299)
    # price=10, volume=100 → ADV = $1_000 (well below $5M floor)
    histories = {
        "PENNY": _make_bars("PENNY", n=300, start_price=10.0, end_price=20.0, volume=100),
    }
    strategy = Momo()
    weights = strategy.target_weights(histories, as_of)
    assert weights == {}
