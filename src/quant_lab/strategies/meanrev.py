"""Short-term mean-reversion strategy.

Entry signal: 5-day cumulative return < -5% AND price > 200-day SMA AND ADV > $10M.
Holds up to 5 names equal-weight.  No news filter in v1.

Position management is stateless: the strategy derives current intended weights
from today's signals.  The paper engine handles rebalancing/drift.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register

_INDEX_PROXIES = {"SPY", "QQQ", "^VIX"}


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
class MeanRev(Strategy):
    """5-day mean reversion in uptrending names (no news filter in v1)."""

    bot_id = "meanrev"
    description = "5-day mean reversion in uptrending names (no news filter in v1)"
    lookback_days: int = 5
    drop_threshold: float = -0.05
    target_uplift: float = 0.03
    max_hold_days: int = 30
    adv_floor: float = 10_000_000
    max_positions: int = 5

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

            # Need at least 200 days for SMA + 5-day window
            if len(eligible) < 205:
                continue

            if _adv(eligible) < self.adv_floor:
                continue

            # 5-day cumulative return
            start_price = eligible[-(self.lookback_days + 1)].close
            end_price = eligible[-1].close
            if start_price <= 0:
                continue
            five_day_return = end_price / start_price - 1.0

            if five_day_return >= self.drop_threshold:
                continue

            # Must be above 200-day SMA (uptrend filter)
            sma200 = _sma(eligible, 200)
            if sma200 is None or end_price <= sma200:
                continue

            signals.append(symbol)

        if not signals:
            return {}

        selected = signals[: self.max_positions]
        weight = 1.0 / len(selected)
        return {sym: weight for sym in selected}
