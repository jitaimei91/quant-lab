"""Tests for MACross — 50/200 golden cross strategy."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.strategies.ma_cross import MACross
from quant_lab.types import Bar

_START = date(2023, 1, 2)
_ADV_VOLUME = 200_000  # price=100, volume=200_000 → ADV=$20M >> $5M floor


def _make_bars(
    symbol: str,
    n: int,
    prices: list[float] | None = None,
    base_price: float = 100.0,
    volume: int = _ADV_VOLUME,
) -> list[Bar]:
    bars = []
    for i in range(n):
        price = prices[i] if prices is not None else base_price
        bars.append(
            Bar(
                symbol=symbol,
                date=_START + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=volume,
            )
        )
    return bars


def test_golden_cross_selected():
    """50d SMA > 200d SMA and price > 50d SMA → selected."""
    n = 210
    # First 160 bars at 80 (pulls 200d SMA down), next 49 bars at 120, final bar at 130.
    # 50d SMA = (49*120 + 130) / 50 = 121.0; 200d SMA = (160*80 + 49*120 + 130) / 210 ≈ 90.1
    # price = 130 > 50d SMA = 121.0; 50d SMA = 121.0 > 200d SMA ≈ 90.1 ✓
    prices = [80.0] * 160 + [120.0] * 49 + [130.0]
    bars = _make_bars("GOLDEN", n=n, prices=prices)
    as_of = _START + timedelta(days=n - 1)
    strategy = MACross()
    weights = strategy.target_weights({"GOLDEN": bars}, as_of)
    assert "GOLDEN" in weights


def test_death_cross_not_selected():
    """50d SMA < 200d SMA (death cross) → not selected."""
    n = 210
    # First 160 bars at 120 (high 200d SMA), then 50 bars at 80 (low 50d SMA)
    prices = [120.0] * 160 + [80.0] * 50
    bars = _make_bars("DEAD", n=n, prices=prices)
    as_of = _START + timedelta(days=n - 1)
    strategy = MACross()
    weights = strategy.target_weights({"DEAD": bars}, as_of)
    assert "DEAD" not in weights


def test_insufficient_history_returns_empty():
    """Fewer than 200 bars → returns {}."""
    n = 150
    bars = _make_bars("SHORT", n=n, base_price=100.0)
    as_of = _START + timedelta(days=n - 1)
    strategy = MACross()
    weights = strategy.target_weights({"SHORT": bars}, as_of)
    assert weights == {}


def test_weights_sum_within_cash_buffer():
    """Sum of weights <= 0.95 (5% cash buffer)."""
    n = 210
    prices = [80.0] * 160 + [120.0] * 49 + [130.0]
    signals = {
        f"STOCK{i}": _make_bars(f"STOCK{i}", n=n, prices=prices)
        for i in range(4)
    }
    as_of = _START + timedelta(days=n - 1)
    strategy = MACross()
    weights = strategy.target_weights(signals, as_of)
    assert sum(weights.values()) <= 0.95 + 1e-9
