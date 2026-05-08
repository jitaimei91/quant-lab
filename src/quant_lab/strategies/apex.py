"""Apex bot: leveraged risk-parity + VRP overlay + VIX kill switch.

Targets backtest Sharpe ~2 via three layered sources of return:

1. **Leveraged risk-parity sleeve** — inverse-vol weighted SSO/TMF/UGL
   (2x S&P, 3x long bonds, 2x gold). The leverage comes from the ETFs
   themselves; the engine only allocates capital normally.

2. **Vol risk premium overlay** — 20% allocation to SVXY (inverse-VIX
   ETF) when the regime is calm (VIX < 15). Historically the highest-
   Sharpe legitimate carry trade for retail; also the most prone to
   catastrophic blowups (XIV, Feb 2018, -90% in one day).

3. **Hard regime kill switch** — uses the existing VIX wiring:
       VIX <  15  → leveraged RP + 20% SVXY (calm: full carry)
       VIX 15-25  → leveraged RP only       (normal: leverage on)
       VIX 25-35  → unleveraged SPY/TLT/GLD (caution: deleverage)
       VIX >= 35  → 100% SHY (panic: cash proxy)

Honest caveats — read before deploying live:
- Leveraged ETFs (2x/3x daily-rebalanced) suffer **volatility decay** in
  choppy markets. 2x SPY does NOT equal 2× annual SPY return — over long
  horizons it can lose to plain SPY in sideways tape.
- SVXY blew up -90% on 5 Feb 2018. Even with 0.5x exposure post-blowup
  and a kill switch, this strategy can lose 30-50% in a single day if VIX
  spikes between checks.
- Backtest Sharpe 2+ is doable; live Sharpe will be ~0.5-1.0 lower due to
  slippage, regime shifts, behavioral fatigue, and gap risk overnight.
- The universe must include SSO/TMF/UGL/SVXY/SHY/^VIX for full effect.
  When missing (e.g., 2008 stress: SSO existed, TMF/UGL/SVXY didn't),
  the strategy degrades gracefully to unleveraged risk parity.
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252

# Component sleeves: leveraged candidate first, fallback last
_STOCK_LEG = ("SSO", "SPY")   # 2x S&P → SPY
_BOND_LEG = ("TMF", "TLT")    # 3x long bonds → TLT
_GOLD_LEG = ("UGL", "GLD")    # 2x gold → GLD
_VRP_SYMBOL = "SVXY"          # inverse-VIX (vol carry)
_DEFENSIVE_SYMBOL = "SHY"     # 1-3yr Treasuries (cash proxy)
_VIX_SYMBOL = "^VIX"

_VOL_WINDOW = 60
_GROSS_LEVERAGE_TARGET = 0.95   # 5% cash buffer
_VRP_ALLOCATION = 0.20          # 20% to SVXY when calm

# Regime thresholds (independent of engine.regime defaults — apex is more
# aggressive about cutting risk than the global PANIC level since leveraged
# ETFs amplify whatever the engine sees)
_VIX_CALM = 15.0
_VIX_CAUTION = 25.0
_VIX_PANIC = 35.0


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


def _resolve_leg(
    histories: dict[str, list[Bar]],
    as_of: date,
    candidates: tuple[str, ...],
) -> tuple[str, list[Bar]] | None:
    """First candidate with enough history at `as_of`; None if all are missing."""
    for sym in candidates:
        bars = [b for b in histories.get(sym, []) if b.date <= as_of]
        if len(bars) > _VOL_WINDOW:
            return sym, bars
    return None


def _inverse_vol_weights(legs: list[tuple[str, list[Bar]]]) -> dict[str, float]:
    """Per-symbol weights ∝ 1/realized_vol, scaled to _GROSS_LEVERAGE_TARGET."""
    inv: dict[str, float] = {}
    for sym, bars in legs:
        vol = _realized_vol(bars, window=_VOL_WINDOW)
        if vol is None or vol <= 0:
            continue
        inv[sym] = 1.0 / vol
    if not inv:
        return {}
    total = sum(inv.values())
    return {sym: _GROSS_LEVERAGE_TARGET * (v / total) for sym, v in inv.items()}


def _resolve_unleveraged(
    histories: dict[str, list[Bar]],
    as_of: date,
) -> dict[str, float]:
    """Inverse-vol weights on the unleveraged sleeves (SPY/TLT/GLD)."""
    legs: list[tuple[str, list[Bar]]] = []
    for candidates in (_STOCK_LEG, _BOND_LEG, _GOLD_LEG):
        unlev_sym = candidates[-1]
        bars = [b for b in histories.get(unlev_sym, []) if b.date <= as_of]
        if len(bars) > _VOL_WINDOW:
            legs.append((unlev_sym, bars))
    return _inverse_vol_weights(legs)


@register
class Apex(Strategy):
    """Leveraged risk-parity + VRP overlay + VIX kill switch (target ~2 Sharpe).

    DO NOT RUN LIVE WITHOUT INDEPENDENT VALIDATION. See module docstring
    for the full caveat list — leveraged-ETF decay and SVXY tail risk are
    real and have wiped out funds before.
    """

    bot_id = "apex"
    description = "Leveraged risk-parity + vol carry + VIX kill switch (target ~2 Sharpe — see caveats)"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        # Read VIX for regime gating; missing VIX → assume NORMAL (no kill switch)
        vix_bars = [b for b in histories.get(_VIX_SYMBOL, []) if b.date <= as_of]
        vix = vix_bars[-1].close if vix_bars else _VIX_CALM + 1  # default to NORMAL

        # PANIC: 100% defensive (or empty if SHY unavailable — runner falls to cash)
        if vix >= _VIX_PANIC:
            shy_bars = [b for b in histories.get(_DEFENSIVE_SYMBOL, []) if b.date <= as_of]
            if shy_bars:
                return {_DEFENSIVE_SYMBOL: _GROSS_LEVERAGE_TARGET}
            return {}

        # CAUTION: deleverage to plain SPY/TLT/GLD risk parity
        if vix >= _VIX_CAUTION:
            return _resolve_unleveraged(histories, as_of)

        # NORMAL or CALM: leveraged risk parity, leveraged sleeves preferred
        legs: list[tuple[str, list[Bar]]] = []
        for candidates in (_STOCK_LEG, _BOND_LEG, _GOLD_LEG):
            leg = _resolve_leg(histories, as_of, candidates)
            if leg is not None:
                legs.append(leg)

        weights = _inverse_vol_weights(legs)
        if not weights:
            return {}

        # CALM: add VRP overlay (cap 20% in SVXY, scale rest down)
        if vix < _VIX_CALM:
            svxy_bars = [b for b in histories.get(_VRP_SYMBOL, []) if b.date <= as_of]
            if len(svxy_bars) > _VOL_WINDOW:
                scale = 1.0 - _VRP_ALLOCATION
                weights = {sym: w * scale for sym, w in weights.items()}
                weights[_VRP_SYMBOL] = _VRP_ALLOCATION * _GROSS_LEVERAGE_TARGET

        return weights
