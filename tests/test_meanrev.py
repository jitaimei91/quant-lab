"""Tests for MeanRev — 5-day mean reversion in uptrend strategy."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.strategies.meanrev import MeanRev
from quant_lab.types import Bar


_START = date(2023, 1, 2)
_VOLUME = 2_000_000  # price * volume = ADV; close=100 → ADV=$200M >> $10M floor


def _make_bars(
    symbol: str,
    n: int,
    base_price: float = 120.0,
    recent_drop: float = 0.0,
    lookback_days: int = 5,
    volume: int = _VOLUME,
) -> list[Bar]:
    """Build `n` bars at `base_price`, then apply `recent_drop` over the final
    `lookback_days` bars.  Prices stay above the 200-day SMA (base_price) so the
    uptrend condition is satisfied unless caller explicitly sets base_price lower
    than the final price.
    """
    bars = []
    for i in range(n):
        days_from_end = n - 1 - i
        if days_from_end < lookback_days:
            # Apply the drop linearly in the final window
            frac = (lookback_days - days_from_end) / lookback_days
            price = base_price * (1.0 + recent_drop * frac)
        else:
            price = base_price
        bars.append(
            Bar(
                symbol=symbol,
                date=_START + timedelta(days=i),
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=volume,
            )
        )
    return bars


def test_drop_in_uptrend_selects_ticker():
    """7% drop over 5 days while price stays above 200-day SMA → selected."""
    # Use 300 bars so we have >205 eligible bars.
    # base_price=120: 200-day SMA ≈ 120; final price ≈ 120 * (1 - 0.07) ≈ 111.6
    # That's BELOW the SMA, so we need final price above base.
    # Better: keep 200-day SMA at ~100 and drop from 120 to 112 (still above 100).
    # Build: first 200 bars at 100, last 100 bars ramp up to 120, final 5 drop to 111.
    bars = []
    n = 310
    for i in range(n):
        days_from_end = n - 1 - i
        if days_from_end < 5:
            # Final 5 bars: drop 7%
            frac = (5 - days_from_end) / 5
            price = 120.0 * (1.0 - 0.07 * frac)
        elif days_from_end < 105:
            # Ramp from 100 → 120 over 100 bars before the drop
            frac = (105 - days_from_end) / 100
            price = 100.0 + frac * 20.0
        else:
            price = 100.0
        bars.append(
            Bar(
                symbol="DIPPED",
                date=_START + timedelta(days=i),
                open=price, high=price * 1.005, low=price * 0.995, close=price,
                volume=_VOLUME,
            )
        )
    as_of = _START + timedelta(days=n - 1)
    histories = {"DIPPED": bars}
    strategy = MeanRev()
    weights = strategy.target_weights(histories, as_of)
    assert "DIPPED" in weights


def test_drop_below_sma_not_selected():
    """5-day drop but price is below 200-day SMA → not selected (downtrend)."""
    # All 300 bars start at 120 and drop to 80 by end → final price well below SMA
    bars = []
    n = 300
    for i in range(n):
        price = 120.0 - (i / (n - 1)) * 60.0  # 120 → 60
        bars.append(
            Bar(
                symbol="DUMPER",
                date=_START + timedelta(days=i),
                open=price, high=price * 1.005, low=price * 0.995, close=price,
                volume=_VOLUME,
            )
        )
    as_of = _START + timedelta(days=n - 1)
    histories = {"DUMPER": bars}
    strategy = MeanRev()
    weights = strategy.target_weights(histories, as_of)
    assert "DUMPER" not in weights


def test_no_signals_returns_empty():
    """Ticker that hasn't dropped → returns {}."""
    n = 300
    bars = _make_bars("STABLE", n=n, base_price=100.0, recent_drop=0.0)
    as_of = _START + timedelta(days=n - 1)
    histories = {"STABLE": bars}
    strategy = MeanRev()
    weights = strategy.target_weights(histories, as_of)
    assert weights == {}


def test_index_proxies_never_selected():
    """SPY / QQQ / ^VIX should never appear in output even with big drops."""
    n = 300
    # SPY with huge drop — still should be excluded
    spy_bars = _make_bars("SPY", n=n, base_price=120.0, recent_drop=-0.20)
    as_of = _START + timedelta(days=n - 1)
    histories = {"SPY": spy_bars}
    strategy = MeanRev()
    weights = strategy.target_weights(histories, as_of)
    assert "SPY" not in weights
