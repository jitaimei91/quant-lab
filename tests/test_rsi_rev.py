"""Tests for RSIRev — RSI<30 reversal strategy (negative control)."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.strategies.rsi_rev import RSIRev, _rsi
from quant_lab.types import Bar

_START = date(2023, 1, 2)
_ADV_VOLUME = 200_000  # price=100, volume=200_000 → ADV=$20M >> $10M floor


def _make_bars(
    symbol: str,
    prices: list[float],
    volume: int = _ADV_VOLUME,
) -> list[Bar]:
    bars = []
    for i, price in enumerate(prices):
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


def _oversold_prices(n: int = 30, base: float = 100.0, drop: float = 0.03) -> list[float]:
    """Generate prices that produce RSI < 30: sustained decline followed by a small uptick."""
    prices = [base] * 5  # initial stable
    # Consecutive down days to drive RSI below 30
    p = base
    for _ in range(n - 6):
        p = p * (1.0 - drop)
        prices.append(p)
    # Confirmation candle: close > yesterday
    prices.append(p * 1.005)
    return prices


def test_rsi_oversold_with_confirmation_selected():
    """Prices that drive RSI<30 with an up confirmation candle → selected."""
    prices = _oversold_prices(n=30, base=100.0, drop=0.04)
    bars = _make_bars("OVERSOLD", prices=prices)
    as_of = _START + timedelta(days=len(prices) - 1)
    strategy = RSIRev()
    weights = strategy.target_weights({"OVERSOLD": bars}, as_of)
    assert "OVERSOLD" in weights


def test_rsi_calculation_matches_reference():
    """_rsi() returns a value in [0, 100] and gives 100 for all-up data."""
    # All-up sequence → RSI = 100
    closes = [100.0 + i for i in range(20)]
    result = _rsi(closes, period=14)
    assert result == 100.0

    # Mixed → value in (0, 100)
    mixed = [100.0, 99.0, 101.0, 98.0, 102.0, 97.0, 103.0, 96.0, 104.0,
             95.0, 105.0, 94.0, 106.0, 93.0, 107.0]
    result2 = _rsi(mixed, period=14)
    assert result2 is not None
    assert 0.0 <= result2 <= 100.0


def test_rsi_overbought_not_selected():
    """When RSI > 70, symbol not selected (would be an exit signal)."""
    # Steady uptrend → RSI > 70
    prices = [90.0 + i * 1.5 for i in range(30)]
    bars = _make_bars("OVERBOUGHT", prices=prices)
    as_of = _START + timedelta(days=len(prices) - 1)
    strategy = RSIRev()
    weights = strategy.target_weights({"OVERBOUGHT": bars}, as_of)
    assert "OVERBOUGHT" not in weights


def test_weights_sum_within_cash_buffer():
    """Sum of weights <= 0.90 (10% cash buffer)."""
    prices = _oversold_prices(n=30, base=100.0, drop=0.04)
    signals = {
        f"OS{i}": _make_bars(f"OS{i}", prices=prices)
        for i in range(2)
    }
    as_of = _START + timedelta(days=len(prices) - 1)
    strategy = RSIRev()
    weights = strategy.target_weights(signals, as_of)
    if weights:
        assert sum(weights.values()) <= 0.90 + 1e-9
