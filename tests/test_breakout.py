"""Tests for Breakout — 52-week high on volume strategy."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.strategies.breakout import Breakout
from quant_lab.types import Bar

_START = date(2023, 1, 2)
_ADV_VOLUME = 200_000  # price=100, volume=200_000 → ADV=$20M >> $5M floor


def _make_bars(
    symbol: str,
    n: int,
    prices: list[float] | None = None,
    volumes: list[int] | None = None,
    base_price: float = 100.0,
    base_volume: int = _ADV_VOLUME,
) -> list[Bar]:
    """Build `n` bars. Optionally provide explicit price/volume lists."""
    bars = []
    for i in range(n):
        price = prices[i] if prices is not None else base_price
        volume = volumes[i] if volumes is not None else base_volume
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


def test_52w_high_with_volume_spike_selected():
    """Monotonic-rising prices at all-time high + volume spike → selected."""
    n = 260
    # Prices rise steadily so the last bar is at the 52-week high
    prices = [100.0 + i * 0.1 for i in range(n)]
    # Volume: last bar is 2x the 20-day avg (1.5x threshold)
    volumes = [_ADV_VOLUME] * (n - 1) + [int(_ADV_VOLUME * 2.0)]
    bars = _make_bars("RISER", n=n, prices=prices, volumes=volumes)
    as_of = _START + timedelta(days=n - 1)
    strategy = Breakout()
    weights = strategy.target_weights({"RISER": bars}, as_of)
    assert "RISER" in weights


def test_same_prices_flat_volume_not_selected():
    """52-week high but flat volume (not a spike) → not selected."""
    n = 260
    prices = [100.0 + i * 0.1 for i in range(n)]
    # Volume stays flat — today's volume == 1.0x avg, below 1.5x threshold
    volumes = [_ADV_VOLUME] * n
    bars = _make_bars("RISER_FLAT_VOL", n=n, prices=prices, volumes=volumes)
    as_of = _START + timedelta(days=n - 1)
    strategy = Breakout()
    weights = strategy.target_weights({"RISER_FLAT_VOL": bars}, as_of)
    assert "RISER_FLAT_VOL" not in weights


def test_insufficient_history_returns_empty():
    """Fewer than 252 bars → returns {}."""
    n = 100
    bars = _make_bars("SHORT", n=n, base_price=100.0)
    as_of = _START + timedelta(days=n - 1)
    strategy = Breakout()
    weights = strategy.target_weights({"SHORT": bars}, as_of)
    assert weights == {}


def test_weights_sum_within_cash_buffer():
    """Sum of weights <= 0.95 (5% cash buffer)."""
    n = 260
    signals = {}
    for i in range(3):
        sym = f"STOCK{i}"
        prices = [100.0 + j * 0.1 for j in range(n)]
        volumes = [_ADV_VOLUME] * (n - 1) + [int(_ADV_VOLUME * 2.0)]
        signals[sym] = _make_bars(sym, n=n, prices=prices, volumes=volumes)
    as_of = _START + timedelta(days=n - 1)
    strategy = Breakout()
    weights = strategy.target_weights(signals, as_of)
    assert sum(weights.values()) <= 0.95 + 1e-9
