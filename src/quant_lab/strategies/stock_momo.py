"""Stock momentum: top-decile S&P 500 by 6-month return, equal-weighted.

The textbook academic momentum factor portfolio:
  - Universe: ~500 S&P 500 large-caps from config/universe_sp500.txt
  - Rank by trailing 6-month total return
  - Hold the top decile (≈50 names) above a positive-return floor
  - Equal-weight, 95% gross (5% cash buffer)
  - Rebalance daily (the engine calls target_weights every morning) but
    weights move slowly enough that turnover is moderate. A monthly rebalance
    would be more cost-realistic; the lab paper-trades so daily is fine.

Why equal-weight (not inverse-vol or rank-weighted):
  - Academic factor portfolios are equal-weight by convention. Comparable.
  - Inverse-vol weighting on 50 single stocks correlates ~0.99 across names
    (most have realized vol 25-40% annualised), so the inverse-vol step
    barely tilts the portfolio. Equal-weight is honest.
  - Rank-weighted (heavier on #1, lighter on #50) is more aggressive but
    introduces concentration risk inconsistent with momentum-as-diversifier.

Survivorship bias warning: yfinance returns today's listed tickers. A
backtest using the current S&P 500 list does NOT include companies that
were once in the index and got delisted (Sears, Lehman, Bear Stearns,
many dot-com era names). Backtest Sharpes inflated by ~0.3-0.5 from this
alone. Live forward performance won't have the bias.
"""
from __future__ import annotations

from datetime import date

from ..types import Bar
from .base import Strategy, register
from ._universe import STOCK_UNIVERSE


_MOMO_WINDOW = 126           # 6 trading months
_MIN_HISTORY = 252           # require 12 months of bars (filters recent IPOs)
_TOP_DECILE_FRACTION = 0.10
_MIN_PICKS = 10              # always hold at least 10 names if available
_MAX_PICKS = 50              # cap at 50 to limit turnover overhead
_GROSS_LEVERAGE = 0.95
# Defensive fallback when no name has positive 6mo momentum (rare — would
# require a market-wide drawdown). Holding cash via empty dict is simpler
# than IEF since stock_momo is a pure equity sleeve.


def _bars_up_to(histories: dict[str, list[Bar]], symbol: str, as_of: date) -> list[Bar]:
    return [b for b in histories.get(symbol, []) if b.date <= as_of]


def _trailing_return(bars: list[Bar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    end = bars[-1].close
    start = bars[-(window + 1)].close
    if start <= 0:
        return None
    return end / start - 1.0


@register
class StockMomo(Strategy):
    """Top-decile S&P 500 momentum, equal-weighted."""

    bot_id = "stock-momo"
    description = "Top-decile S&P 500 by 6mo return, equal-weighted"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        scored: list[tuple[str, float]] = []
        for sym in STOCK_UNIVERSE:
            bars = _bars_up_to(histories, sym, as_of)
            if len(bars) < _MIN_HISTORY:
                continue
            ret = _trailing_return(bars, _MOMO_WINDOW)
            if ret is None or ret <= 0:
                continue
            scored.append((sym, ret))

        if not scored:
            return {}

        scored.sort(key=lambda x: x[1], reverse=True)

        # Top decile, bounded by [_MIN_PICKS, _MAX_PICKS]
        n_picks = max(_MIN_PICKS, int(round(len(scored) * _TOP_DECILE_FRACTION)))
        n_picks = min(n_picks, _MAX_PICKS, len(scored))
        picks = [sym for sym, _ in scored[:n_picks]]

        per_position = _GROSS_LEVERAGE / n_picks
        return {sym: per_position for sym in picks}
