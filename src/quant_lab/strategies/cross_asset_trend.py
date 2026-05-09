"""Cross-asset trend: simple absolute-momentum baseline across asset classes.

For each leg in {SPY, EFA, EEM, TLT, GLD, USO}:
  - hold the leg only if its 12-month trailing total return is positive
  - inverse-vol weight survivors

This is the GTAA / Faber-style absolute-momentum baseline. Intentionally
simple — no trend filter on top, no leverage, no master kill switches.
Functions as a robustness check and a benchmark for apex's more elaborate
machinery: if apex isn't beating this honestly, all the extra logic
isn't earning its complexity.

Honest expectation: aggregate Sharpe 0.5-0.8 across regimes. Antonacci-style
absolute momentum is the most-replicated cross-asset edge in academia,
even after costs.
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252
_LEGS = ("SPY", "EFA", "EEM", "TLT", "GLD", "USO")
_FALLBACK = "IEF"  # parked here when no leg has positive momentum
_MOMO_WINDOW = 252  # 12 months — Antonacci's documented sweet spot
_VOL_WINDOW = 60
_GROSS_LEVERAGE = 0.95


def _bars_up_to(histories: dict[str, list[Bar]], symbol: str, as_of: date) -> list[Bar]:
    return [b for b in histories.get(symbol, []) if b.date <= as_of]


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


def _trailing_return(bars: list[Bar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    end = bars[-1].close
    start = bars[-(window + 1)].close
    if start <= 0:
        return None
    return end / start - 1.0


@register
class CrossAssetTrend(Strategy):
    """Antonacci-style absolute momentum across SPY/EFA/EEM/TLT/GLD/USO."""

    bot_id = "cross-asset-trend"
    description = "Absolute momentum (12mo) across 6 asset classes, inverse-vol weighted"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        survivors: list[tuple[str, float]] = []
        for sym in _LEGS:
            bars = _bars_up_to(histories, sym, as_of)
            ret = _trailing_return(bars, _MOMO_WINDOW)
            vol = _realized_vol(bars, _VOL_WINDOW)
            if ret is None or vol is None or vol <= 0 or ret <= 0:
                continue
            survivors.append((sym, vol))

        if not survivors:
            if _bars_up_to(histories, _FALLBACK, as_of):
                return {_FALLBACK: _GROSS_LEVERAGE}
            return {}

        inv_vols = {sym: 1.0 / vol for sym, vol in survivors}
        total = sum(inv_vols.values())
        return {sym: _GROSS_LEVERAGE * (v / total) for sym, v in inv_vols.items()}
