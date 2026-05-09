"""50/200 golden-cross strategy.

"Long" condition: 50-day SMA > 200-day SMA (golden cross persists) AND price >
50-day SMA.  ADV > $5M.  Equal-weight up to 10 positions; 5% cash buffer.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register

_INDEX_PROXIES = {"SPY", "QQQ", "^VIX", "SSO", "TMF", "UGL", "SVXY", "SHY", "XLK", "XLY", "XLV", "XLF", "XLP", "XLE", "XLI", "XLU", "XLRE", "XLB", "XLC", "LQD"}  # also exclude apex-only sleeves


def _sma(bars: list[Bar], window: int) -> float | None:
    """Simple moving average of close prices over the last `window` bars."""
    if len(bars) < window:
        return None
    return sum(b.close for b in bars[-window:]) / window


def _adv(bars: list[Bar], window: int = 20) -> float:
    """Average daily dollar volume over the last `window` bars."""
    recent = bars[-window:]
    if not recent:
        return 0.0
    return sum(b.close * b.volume for b in recent) / len(recent)


@register
class MACross(Strategy):
    """50/200 SMA golden cross — equal-weight, max 10 positions."""

    bot_id = "ma_cross"
    description = "50/200 SMA golden cross: 50d > 200d and price > 50d"
    sma_fast: int = 50
    sma_slow: int = 200
    adv_floor: float = 5_000_000
    max_positions: int = 10
    cash_buffer: float = 0.05

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        signals: list[str] = []

        for symbol, bars in histories.items():
            if symbol in _INDEX_PROXIES:
                continue

            eligible = [b for b in bars if b.date <= as_of]

            # Need at least sma_slow bars
            if len(eligible) < self.sma_slow:
                continue

            if _adv(eligible) < self.adv_floor:
                continue

            price = eligible[-1].close
            fast = _sma(eligible, self.sma_fast)
            slow = _sma(eligible, self.sma_slow)

            if fast is None or slow is None:
                continue

            # Golden cross: 50d > 200d AND price above 50d
            if fast <= slow:
                continue
            if price <= fast:
                continue

            signals.append(symbol)

        if not signals:
            return {}

        selected = signals[: self.max_positions]
        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}
