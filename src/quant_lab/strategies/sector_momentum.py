"""Sector momentum: top-3 SPDR sectors by 6-month return, inverse-vol weighted.

US sector dispersion is real — the gap between the best-performing and
worst-performing sector in any given quarter is typically 15-25%. This bot
captures that by ranking the 11 SPDR sector ETFs on trailing 6-month return
and holding the top 3 (positive-return only) inverse-vol weighted.

Why it's NOT a duplicate of apex's dual-momentum:
  - apex picks across asset classes (stocks/bonds/gold/commodities).
    sector_momentum picks WITHIN US equities — it's a market-neutral-ish
    rotation that doesn't fight the apex's broad-asset signal.
  - In bull regimes the two stack: apex says "stay long equities," sector
    rotation picks WHICH equities. In bear regimes apex flips defensive,
    and this bot also goes defensive (no positive 6mo sectors → cash/IEF).

Universe is the 11 SPDR sectors, all with daily liquidity > $100M:
  XLK tech / XLY consumer disc / XLV health / XLF financials / XLP staples
  XLE energy / XLI industrials / XLU utilities / XLRE real estate
  XLB materials / XLC communications
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252
_SECTORS = ("XLK", "XLY", "XLV", "XLF", "XLP", "XLE", "XLI", "XLU", "XLRE", "XLB", "XLC")
_DEFENSIVE_FALLBACK = "IEF"  # treasuries when no sector has positive momentum
_MOMO_WINDOW = 126
_VOL_WINDOW = 60
_TOP_N = 3
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
class SectorMomentum(Strategy):
    """Rotate top-3 SPDR sectors by trailing 6-month return, inverse-vol weighted."""

    bot_id = "sector-momo"
    description = "Top-3 SPDR sectors by 6mo return, inverse-vol weighted (defensive fallback)"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        scored: list[tuple[str, float, float]] = []  # (symbol, momentum, vol)
        for sym in _SECTORS:
            bars = _bars_up_to(histories, sym, as_of)
            ret = _trailing_return(bars, _MOMO_WINDOW)
            vol = _realized_vol(bars, _VOL_WINDOW)
            if ret is None or vol is None or vol <= 0:
                continue
            if ret <= 0:
                continue
            scored.append((sym, ret, vol))

        if not scored:
            # No sector has positive momentum → defensive
            if _bars_up_to(histories, _DEFENSIVE_FALLBACK, as_of):
                return {_DEFENSIVE_FALLBACK: _GROSS_LEVERAGE}
            return {}

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:_TOP_N]

        inv_vols = {sym: 1.0 / vol for sym, _, vol in top}
        total = sum(inv_vols.values())
        return {sym: _GROSS_LEVERAGE * (v / total) for sym, v in inv_vols.items()}
