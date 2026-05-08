"""RSI reversal strategy — negative control.

Entry: 14-period RSI < 30 with confirmation candle (today's close > yesterday's
close).  Exit when RSI > 70 (strategy returns weight 0 for those).
ADV > $10M.  Equal-weight up to 3 positions; 10% cash buffer.

This strategy is included as a negative control — included to demonstrate
weak-evidence strategies losing in the live tournament.
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


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute RSI using simple average gains/losses (matches features.py reference).

    Matches morning_quant_bot.features.rsi() — simple average over a rolling
    window of `period` days rather than Wilder's exponential smoothing.
    """
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1) :]
    gains = 0.0
    losses = 0.0
    for prev, curr in zip(window[:-1], window[1:]):
        change = curr - prev
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


@register
class RSIRev(Strategy):
    """RSI < 30 reversal — negative control — included to demonstrate weak-evidence strategies losing."""

    bot_id = "rsi_rev"
    description = (
        "RSI<30 reversal with confirmation candle — "
        "negative control — included to demonstrate weak-evidence strategies losing"
    )
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    adv_floor: float = 10_000_000
    max_positions: int = 3
    cash_buffer: float = 0.10

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

            # Need at least rsi_period + 2 bars (period for RSI + 1 for confirmation candle)
            if len(eligible) < self.rsi_period + 2:
                continue

            if _adv(eligible) < self.adv_floor:
                continue

            closes = [b.close for b in eligible]
            current_rsi = _rsi(closes, self.rsi_period)
            if current_rsi is None:
                continue

            # RSI > 70: not selected (return 0 weight for this symbol)
            if current_rsi > self.rsi_overbought:
                continue

            # RSI < 30 with confirmation candle: today's close > yesterday's close
            if current_rsi < self.rsi_oversold:
                today_close = eligible[-1].close
                yesterday_close = eligible[-2].close
                if today_close > yesterday_close:
                    signals.append(symbol)

        if not signals:
            return {}

        selected = signals[: self.max_positions]
        weight = (1.0 - self.cash_buffer) / len(selected)
        return {sym: weight for sym in selected}
