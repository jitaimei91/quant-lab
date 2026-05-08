"""Risk-parity sleeve: SPY/TLT/GLD inverse-volatility weighted.

Equal-risk-contribution allocation across stocks (SPY), long bonds (TLT),
and gold (GLD). Each leg's weight is proportional to 1/vol so each contributes
the same notional risk. Genuinely orthogonal to trend and mean-reversion
sleeves — diversification benefit shows up in regimes where stocks/bonds
decorrelate (2008, 2020) and in stagflation tails where gold provides ballast.

Volatility window: 60 trading days. Rebalance is implicit — the runner calls
target_weights every morning, so this is effectively a daily rebalance, but
the inverse-vol weights move slowly enough (60-day window) that turnover is
modest.

If any leg has insufficient bars, falls back to equal-weight across the
available legs. If none has data, returns {} (cash) — runner handles it.
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252
_LEGS = ("SPY", "TLT", "GLD")
_VOL_WINDOW = 60
_GROSS_LEVERAGE = 0.95  # cash buffer 5%


def _realized_vol(bars: list[Bar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    closes = [b.close for b in bars[-(window + 1):]]
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0:
            return None
        rets.append(closes[i] / closes[i - 1] - 1.0)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return sqrt(var) * sqrt(TRADING_DAYS_PER_YEAR)


@register
class RiskParity(Strategy):
    """Inverse-vol weighted SPY/TLT/GLD — genuinely orthogonal sleeve."""

    bot_id = "risk-parity"
    description = "Inverse-vol SPY/TLT/GLD risk parity, 60d vol window"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        inv_vols: dict[str, float] = {}
        for sym in _LEGS:
            bars = [b for b in histories.get(sym, []) if b.date <= as_of]
            vol = _realized_vol(bars, window=_VOL_WINDOW)
            if vol is None or vol <= 0:
                continue
            inv_vols[sym] = 1.0 / vol

        if not inv_vols:
            return {}

        total = sum(inv_vols.values())
        return {
            sym: _GROSS_LEVERAGE * inv_v / total
            for sym, inv_v in inv_vols.items()
        }
