"""Cross-sectional 6-month momentum strategy.

Ranks all tradable tickers in `histories` by trailing 126-day return, selects
the top decile (or top 10, whichever is smaller), and equal-weights them with a
5% cash buffer.  Index proxies (SPY, QQQ, ^VIX) are excluded from selection.
ADV and minimum history filters are applied before ranking.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register

_INDEX_PROXIES = {"SPY", "QQQ", "^VIX"}


def _adv(bars: list[Bar], window: int = 20) -> float:
    """Average daily dollar volume over the last `window` bars."""
    recent = bars[-window:]
    if not recent:
        return 0.0
    return sum(b.close * b.volume for b in recent) / len(recent)


@register
class Momo(Strategy):
    """Cross-sectional 6-month momentum, top decile, equal-weight."""

    bot_id = "momo"
    description = "Cross-sectional 6-month momentum, top decile, equal-weight"
    lookback_days: int = 126
    cash_buffer: float = 0.05
    adv_floor: float = 5_000_000
    min_history_days: int = 250

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        scores: list[tuple[float, str]] = []

        for symbol, bars in histories.items():
            if symbol in _INDEX_PROXIES:
                continue

            # Restrict to data on or before as_of
            eligible = [b for b in bars if b.date <= as_of]

            if len(eligible) < self.min_history_days:
                continue

            if _adv(eligible) < self.adv_floor:
                continue

            # Trailing return over lookback_days
            start_bar = eligible[-(self.lookback_days + 1)]
            end_bar = eligible[-1]
            if start_bar.close <= 0:
                continue
            trailing_return = end_bar.close / start_bar.close - 1.0
            scores.append((trailing_return, symbol))

        if not scores:
            return {}

        scores.sort(reverse=True)

        n_universe = len(scores)
        top_n = max(1, min(10, n_universe // 10))
        selected = [sym for _, sym in scores[:top_n]]

        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}
