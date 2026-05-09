"""Credit carry: harvest credit spread when risk-on, flight-to-quality when risk-off.

Mechanism:
  - SPY > 200d MA  →  hold 50% HYG (high yield) + 50% LQD (investment grade).
    HYG/LQD pay a coupon premium over treasuries that compresses (price
    rises) when default risk is low — i.e., the credit-carry trade.
  - SPY <= 200d MA →  hold 100% IEF (7-10yr treasuries). Flight-to-quality
    rally typically accompanies equity drawdowns.

Why this is honest carry, not curve-fitting:
  - Credit spreads have a real economic basis (default-risk compensation)
    that has paid out over decades. The carry is small (~2-4% annualised
    over treasuries) but very stable in expansions.
  - The trend filter (SPY 200d MA) is the standard regime indicator —
    using a more elaborate one would risk overfitting historical regime
    boundaries.

Honest expectations: standalone Sharpe 0.4-0.7 in expansions, can lose
modestly during default cycles (2008 HYG was -25% intra-year before
recovering). The trend filter mitigates that, but doesn't eliminate it.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register


_RISK_ON_LEGS = ("HYG", "LQD")
_RISK_OFF_LEG = "IEF"
_FALLBACK_LEG = "SHY"  # if neither IEF nor HYG/LQD have history
_TREND_WINDOW = 200
_GROSS_LEVERAGE = 0.95


def _bars_up_to(histories: dict[str, list[Bar]], symbol: str, as_of: date) -> list[Bar]:
    return [b for b in histories.get(symbol, []) if b.date <= as_of]


def _spy_uptrend(spy_bars: list[Bar]) -> bool:
    if len(spy_bars) < _TREND_WINDOW:
        return True  # bullish default when history is short
    closes = [b.close for b in spy_bars[-_TREND_WINDOW:]]
    ma = sum(closes) / _TREND_WINDOW
    return spy_bars[-1].close > ma


@register
class CreditCarry(Strategy):
    """50/50 HYG/LQD risk-on, 100% IEF risk-off (SPY 200d MA)."""

    bot_id = "credit-carry"
    description = "Credit-carry sleeve: HYG+LQD when SPY > 200d MA, IEF otherwise"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        spy_bars = _bars_up_to(histories, "SPY", as_of)
        risk_on = _spy_uptrend(spy_bars)

        if risk_on:
            available = [
                sym for sym in _RISK_ON_LEGS
                if _bars_up_to(histories, sym, as_of)
            ]
            if available:
                per_leg = _GROSS_LEVERAGE / len(available)
                return {sym: per_leg for sym in available}
            # Risk-on universe missing → fall through to risk-off

        # Risk-off (or risk-on legs missing): hold IEF, then SHY
        for sym in (_RISK_OFF_LEG, _FALLBACK_LEG):
            if _bars_up_to(histories, sym, as_of):
                return {sym: _GROSS_LEVERAGE}
        return {}
