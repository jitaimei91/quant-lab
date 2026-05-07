from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252


def _realized_vol(bars: list[Bar], window: int) -> float | None:
    """Annualized realized volatility from daily log returns over the last `window` bars."""
    if len(bars) <= window:
        return None
    closes = [b.close for b in bars[-(window + 1):]]
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            return None
        rets.append((closes[i] / closes[i - 1]) - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return sqrt(var) * sqrt(TRADING_DAYS_PER_YEAR)


class _VolTargetedIndex(Strategy):
    """Vol-targeted long exposure to a single ETF symbol."""

    symbol: str = ""
    target_vol_default: float = 0.15
    vol_window: int = 60

    def __init__(self, target_vol: float | None = None):
        self.target_vol = target_vol or self.target_vol_default

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        bars = histories.get(self.symbol, [])
        bars = [b for b in bars if b.date <= as_of]
        realized = _realized_vol(bars, window=self.vol_window)
        if realized is None or realized <= 0:
            return {}
        weight = min(1.0, self.target_vol / realized)
        return {self.symbol: weight}


@register
class SPYVol(_VolTargetedIndex):
    bot_id = "spy-vol"
    description = "Vol-targeted long S&P 500 (SPY). Honest market benchmark."
    symbol = "SPY"
