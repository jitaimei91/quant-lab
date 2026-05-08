"""Apex bot v2 — trend-filtered leveraged risk-parity with circuit breakers.

Lessons from v1 stress tests:
- v1 was crushed by Feb 2018 Volmageddon (SVXY -90% intraday). Daily VIX
  kill switch can't react to overnight gaps. **v2 drops SVXY entirely.**
- v1 stayed in leveraged risk parity through 2022 → TMF -50% drag wrecked
  it. **v2 adds an SPY 200-day-MA trend filter** so leveraged exposure
  only fires in confirmed uptrends.
- v1's SHY-only defensive missed flight-to-quality in COVID/2008.
  **v2 uses IEF** (medium-duration treasuries) in panic regimes so we
  capture the bond rally that typically accompanies risk-off.

Decision matrix:
                          | calm     | normal   | caution  | panic
                          | (VIX<15) | (VIX<25) | (VIX<35) | (VIX≥35)
    SPY > 200d MA  (up)   | LEV-RP   | LEV-RP   | UNLEV-RP | IEF 100%
    SPY ≤ 200d MA (down)  | DEF      | DEF      | DEF      | IEF 100%
    SPY 60d DD > 15%      | DEF      | DEF      | DEF      | IEF 100%

  LEV-RP   = inverse-vol on SSO/TMF/UGL (leverage from the ETFs themselves)
  UNLEV-RP = inverse-vol on SPY/TLT/GLD
  DEF      = 50% IEF / 50% SHY  (defensive bond mix; capture flight-to-
             quality when stress arrives without giving up all yield)

The drawdown circuit breaker (15% off the 60-day SPY high) is what
catches fast crashes the 200-day MA misses (e.g., Feb-Mar 2020 COVID).

Honest expected backtest Sharpe: 1.0–1.5 across regimes. Anything higher
on free-tier daily ETF data is overfitting noise.
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252

# Leveraged candidates (preferred when uptrend + low-vol regime)
_STOCK_LEG = ("SSO", "SPY")
_BOND_LEG = ("TMF", "TLT")
_GOLD_LEG = ("UGL", "GLD")
_PANIC_BOND = "IEF"      # medium-duration flight-to-quality
_DEFENSIVE_BOND = "IEF"  # used in defensive 50/50 mix
_DEFENSIVE_CASH = "SHY"  # 1-3yr cash proxy
_VIX_SYMBOL = "^VIX"

_VOL_WINDOW = 60
_TREND_WINDOW = 200
_DD_WINDOW = 60
_DD_CIRCUIT_BREAKER = -0.15  # SPY 15% off 60-day high → defensive
_GROSS_LEVERAGE = 0.95       # 5% cash buffer

# VIX regime thresholds — tighter than the global engine since leveraged
# ETFs amplify whatever the engine sees and we want to deleverage early.
_VIX_CALM = 15.0
_VIX_CAUTION = 25.0
_VIX_PANIC = 35.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _inverse_vol_weights(legs: list[tuple[str, list[Bar]]]) -> dict[str, float]:
    inv: dict[str, float] = {}
    for sym, bars in legs:
        vol = _realized_vol(bars, window=_VOL_WINDOW)
        if vol is None or vol <= 0:
            continue
        inv[sym] = 1.0 / vol
    if not inv:
        return {}
    total = sum(inv.values())
    return {sym: _GROSS_LEVERAGE * (v / total) for sym, v in inv.items()}


def _resolve_first_with_history(
    histories: dict[str, list[Bar]],
    as_of: date,
    candidates: tuple[str, ...],
) -> tuple[str, list[Bar]] | None:
    for sym in candidates:
        bars = _bars_up_to(histories, sym, as_of)
        if len(bars) > _VOL_WINDOW:
            return sym, bars
    return None


def _spy_uptrend(spy_bars: list[Bar]) -> bool:
    """Return True if SPY's last close is above its 200-day moving average."""
    if len(spy_bars) < _TREND_WINDOW:
        return True  # not enough history → assume bullish (won't fire defensive)
    window_closes = [b.close for b in spy_bars[-_TREND_WINDOW:]]
    ma = sum(window_closes) / _TREND_WINDOW
    return spy_bars[-1].close > ma


def _spy_drawdown(spy_bars: list[Bar]) -> float:
    """Return current SPY drawdown vs 60-day rolling high. 0.0 if no data."""
    if len(spy_bars) < 2:
        return 0.0
    window = spy_bars[-_DD_WINDOW:] if len(spy_bars) >= _DD_WINDOW else spy_bars
    peak = max(b.close for b in window)
    if peak <= 0:
        return 0.0
    return spy_bars[-1].close / peak - 1.0


# ---------------------------------------------------------------------------
# Sleeves
# ---------------------------------------------------------------------------


def _leveraged_rp(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    """Inverse-vol on leveraged sleeves with graceful unleveraged fallback."""
    legs: list[tuple[str, list[Bar]]] = []
    for candidates in (_STOCK_LEG, _BOND_LEG, _GOLD_LEG):
        leg = _resolve_first_with_history(histories, as_of, candidates)
        if leg is not None:
            legs.append(leg)
    return _inverse_vol_weights(legs)


def _unleveraged_rp(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    """Inverse-vol on plain SPY/TLT/GLD."""
    legs: list[tuple[str, list[Bar]]] = []
    for candidates in (_STOCK_LEG, _BOND_LEG, _GOLD_LEG):
        unlev = candidates[-1]
        bars = _bars_up_to(histories, unlev, as_of)
        if len(bars) > _VOL_WINDOW:
            legs.append((unlev, bars))
    return _inverse_vol_weights(legs)


def _defensive_mix(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    """50% IEF / 50% SHY — defensive bond mix. Falls back gracefully."""
    legs = [
        sym for sym in (_DEFENSIVE_BOND, _DEFENSIVE_CASH)
        if _bars_up_to(histories, sym, as_of)
    ]
    if not legs:
        # Try TLT as ultimate fallback
        for sym in ("TLT", "SHY", "IEF"):
            if _bars_up_to(histories, sym, as_of):
                return {sym: _GROSS_LEVERAGE}
        return {}
    per_leg = _GROSS_LEVERAGE / len(legs)
    return {sym: per_leg for sym in legs}


def _flight_to_quality(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    """100% IEF in panic uptrend (or fallback to TLT, then SHY)."""
    for sym in (_PANIC_BOND, "TLT", _DEFENSIVE_CASH):
        if _bars_up_to(histories, sym, as_of):
            return {sym: _GROSS_LEVERAGE}
    return {}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@register
class Apex(Strategy):
    """Trend-filtered leveraged risk-parity with drawdown circuit breaker.

    See module docstring for the full decision matrix and the lessons-learned
    from v1 that drove this design.
    """

    bot_id = "apex"
    description = "Trend-filtered leveraged RP + DD circuit breaker (target ~1.2 Sharpe)"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        # Read regime inputs
        vix_bars = _bars_up_to(histories, _VIX_SYMBOL, as_of)
        vix = vix_bars[-1].close if vix_bars else _VIX_CALM + 1  # default NORMAL

        spy_bars = _bars_up_to(histories, "SPY", as_of)
        uptrend = _spy_uptrend(spy_bars)
        drawdown = _spy_drawdown(spy_bars)

        # Panic always overrides — flight-to-quality regardless of trend
        if vix >= _VIX_PANIC:
            return _flight_to_quality(histories, as_of)

        # Drawdown circuit breaker — 15% off the 60-day high triggers defensive
        # even if the long-term trend is still up. Catches fast crashes that
        # the 200-day MA hasn't yet broken.
        if drawdown < _DD_CIRCUIT_BREAKER:
            return _defensive_mix(histories, as_of)

        # Trend filter — never run leveraged exposure in confirmed downtrend
        if not uptrend:
            return _defensive_mix(histories, as_of)

        # Uptrend + caution → deleverage to plain SPY/TLT/GLD
        if vix >= _VIX_CAUTION:
            return _unleveraged_rp(histories, as_of)

        # Uptrend + normal/calm → leveraged risk parity
        return _leveraged_rp(histories, as_of)
