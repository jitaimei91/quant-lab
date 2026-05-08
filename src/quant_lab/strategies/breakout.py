"""52-week high breakout strategy.

Entry signal: today's close equals the maximum close over the last 252 trading
days (52-week high) AND today's volume >= 1.5x the 20-day average volume AND
ADV > $5M.  Equal-weight up to 5 positions; 5% cash buffer.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register

_INDEX_PROXIES = {"SPY", "QQQ", "^VIX", "SSO", "TMF", "UGL", "SVXY", "SHY"}  # also exclude apex-only sleeves


def _adv(bars: list[Bar], window: int = 20) -> float:
    """Average daily dollar volume over the last `window` bars."""
    recent = bars[-window:]
    if not recent:
        return 0.0
    return sum(b.close * b.volume for b in recent) / len(recent)


@register
class Breakout(Strategy):
    """52-week high on volume >= 1.5x 20d avg."""

    bot_id = "breakout"
    description = "52-week high on volume >= 1.5x 20d avg"
    high_lookback: int = 252
    volume_lookback: int = 20
    volume_multiplier: float = 1.5
    adv_floor: float = 5_000_000
    max_positions: int = 5
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

            if len(eligible) < self.high_lookback:
                continue

            if _adv(eligible) < self.adv_floor:
                continue

            today_bar = eligible[-1]
            today_close = today_bar.close
            today_volume = today_bar.volume

            # 52-week high check: today's close must equal the max of last 252 closes
            recent_closes = [b.close for b in eligible[-self.high_lookback :]]
            if today_close < max(recent_closes):
                continue

            # Volume confirmation: today's volume >= 1.5x 20-day average volume
            avg_volume = sum(b.volume for b in eligible[-self.volume_lookback :]) / self.volume_lookback
            if today_volume < self.volume_multiplier * avg_volume:
                continue

            signals.append(symbol)

        if not signals:
            return {}

        selected = signals[: self.max_positions]
        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}
