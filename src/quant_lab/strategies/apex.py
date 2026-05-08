"""Apex bot v3 — dual momentum + portfolio vol-targeting + master switches.

Layered design (read top-down — first true gate routes the bot):

    1.  vix >= 35           → flight-to-quality (100% IEF)
    2.  spy 60d DD < -15%   → defensive (50% IEF / 50% SHY)
    3.  spy < 200d MA       → defensive
    4.  vix >= 25 (caution) → unleveraged-only top-3 dual momentum
    5.  otherwise           → top-3 dual momentum, leveraged where allowed
        (calm + normal)       and vol-targeted to 12% annualized portfolio vol

Key v3 changes over v2:

- **Dual momentum across a diversified universe.** Rank ETFs by 6-month
  total return; pick the top 3 with positive returns; inverse-vol weight.
  Universe: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, USO, VNQ. This avoids
  the v2 problem of being permanently long SPY+TLT+GLD even when the
  best performers are e.g. QQQ + GLD + EEM. Antonacci-style.

- **Portfolio-level vol targeting.** Compute realized portfolio vol from
  weighted leg vols (treating legs as independent — first-order approx),
  then scale gross exposure so portfolio annualised vol ≈ 12%. Below
  target → up to 95% gross. Above → scale down. This is what smooths
  the leveraged-ETF-decay drag in choppy regimes that hit v2.

- **Leverage gates per leg.** A leveraged ETF (SSO/TMF/UGL) is used in
  place of its unleveraged sibling only when (i) regime is normal/calm,
  (ii) trend is up, and (iii) the leg has at least 60 bars of history.

Honest expected backtest Sharpe across stress regimes: 1.2–1.5 aggregate.
Live will land 0.5–1.0 lower. Sharpe 2 still needs paid intraday/options data.
"""
from __future__ import annotations

from datetime import date
from math import sqrt

from ..types import Bar
from .base import Strategy, register


TRADING_DAYS_PER_YEAR = 252

# Dual-momentum universe (unleveraged baselines)
_MOMO_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "USO", "VNQ")

# Per-leg leverage upgrade map. When the regime allows leverage AND the
# leveraged ETF has enough history, swap the unleveraged pick for its 2x/3x
# sibling. Anything not in this dict stays unleveraged.
_LEVERAGE_UPGRADE = {
    "SPY": "SSO",   # 2x S&P
    "TLT": "TMF",   # 3x long bonds
    "GLD": "UGL",   # 2x gold
}

_PANIC_BOND = "IEF"
_DEFENSIVE_BOND = "IEF"
_DEFENSIVE_CASH = "SHY"
_VIX_SYMBOL = "^VIX"

# Lookback windows
_MOMO_WINDOW = 126     # 6 trading months
_VOL_WINDOW = 60
_TREND_WINDOW = 200
_DD_WINDOW = 60

# Risk knobs
_DD_CIRCUIT_BREAKER = -0.15
_TARGET_PORTFOLIO_VOL = 0.12      # 12% annualised
_GROSS_LEVERAGE_CAP = 0.95
_TOP_N = 3                        # how many momentum picks to hold

# VIX regime thresholds (apex-internal, more conservative than engine.regime)
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


def _trailing_return(bars: list[Bar], window: int) -> float | None:
    if len(bars) <= window:
        return None
    end = bars[-1].close
    start = bars[-(window + 1)].close
    if start <= 0:
        return None
    return end / start - 1.0


def _spy_uptrend(spy_bars: list[Bar]) -> bool:
    if len(spy_bars) < _TREND_WINDOW:
        return True
    window_closes = [b.close for b in spy_bars[-_TREND_WINDOW:]]
    ma = sum(window_closes) / _TREND_WINDOW
    return spy_bars[-1].close > ma


def _spy_drawdown(spy_bars: list[Bar]) -> float:
    if len(spy_bars) < 2:
        return 0.0
    window = spy_bars[-_DD_WINDOW:] if len(spy_bars) >= _DD_WINDOW else spy_bars
    peak = max(b.close for b in window)
    if peak <= 0:
        return 0.0
    return spy_bars[-1].close / peak - 1.0


# ---------------------------------------------------------------------------
# Core sleeves
# ---------------------------------------------------------------------------


def _flight_to_quality(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    for sym in (_PANIC_BOND, "TLT", _DEFENSIVE_CASH):
        if _bars_up_to(histories, sym, as_of):
            return {sym: _GROSS_LEVERAGE_CAP}
    return {}


def _defensive_mix(histories: dict[str, list[Bar]], as_of: date) -> dict[str, float]:
    legs = [
        sym for sym in (_DEFENSIVE_BOND, _DEFENSIVE_CASH)
        if _bars_up_to(histories, sym, as_of)
    ]
    if not legs:
        for sym in ("TLT", "SHY", "IEF"):
            if _bars_up_to(histories, sym, as_of):
                return {sym: _GROSS_LEVERAGE_CAP}
        return {}
    per_leg = _GROSS_LEVERAGE_CAP / len(legs)
    return {sym: per_leg for sym in legs}


def _dual_momentum_picks(
    histories: dict[str, list[Bar]],
    as_of: date,
) -> list[str]:
    """Top-N unleveraged picks by 6-month return, requiring positive return.

    Returns symbols from _MOMO_UNIVERSE only; the leverage-upgrade happens
    in the caller after gating on regime.
    """
    scored: list[tuple[str, float]] = []
    for sym in _MOMO_UNIVERSE:
        bars = _bars_up_to(histories, sym, as_of)
        ret = _trailing_return(bars, _MOMO_WINDOW)
        vol = _realized_vol(bars, _VOL_WINDOW)
        if ret is None or vol is None or vol <= 0:
            continue
        if ret <= 0:
            continue  # Antonacci's "no edge" gate — refuse to hold losers
        scored.append((sym, ret))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scored[:_TOP_N]]


def _vol_targeted_inverse_vol(
    histories: dict[str, list[Bar]],
    as_of: date,
    symbols: list[str],
    target_vol: float,
) -> dict[str, float]:
    """Inverse-vol weight `symbols`, then scale gross to hit target portfolio vol.

    Approximates portfolio variance as the sum of (w_i * sigma_i)^2 — which
    assumes zero pairwise correlation. The error is small for diversified
    asset-class sleeves and avoids needing a full covariance matrix that
    would overfit on rolling 60-day windows.

    Result is capped at _GROSS_LEVERAGE_CAP regardless of the vol target.
    """
    inv_vols: dict[str, float] = {}
    leg_vols: dict[str, float] = {}
    for sym in symbols:
        bars = _bars_up_to(histories, sym, as_of)
        vol = _realized_vol(bars, window=_VOL_WINDOW)
        if vol is None or vol <= 0:
            continue
        inv_vols[sym] = 1.0 / vol
        leg_vols[sym] = vol
    if not inv_vols:
        return {}

    total_inv = sum(inv_vols.values())
    base_weights = {sym: v / total_inv for sym, v in inv_vols.items()}

    # Approximate portfolio vol assuming uncorrelated legs
    port_var = sum((w * leg_vols[sym]) ** 2 for sym, w in base_weights.items())
    port_vol = sqrt(port_var) if port_var > 0 else 0.0

    if port_vol <= 0:
        return {sym: w * _GROSS_LEVERAGE_CAP for sym, w in base_weights.items()}

    # Scale gross to hit target vol; cap at gross-leverage limit
    scale = min(target_vol / port_vol, _GROSS_LEVERAGE_CAP / 1.0)
    return {sym: w * scale for sym, w in base_weights.items()}


def _maybe_upgrade_to_leveraged(
    weights: dict[str, float],
    histories: dict[str, list[Bar]],
    as_of: date,
) -> dict[str, float]:
    """For each pick that has a leveraged sibling with sufficient history,
    swap it. Halves the unleveraged weight first since the leveraged ETF
    already provides 2-3x exposure.
    """
    out: dict[str, float] = {}
    for sym, w in weights.items():
        upgrade = _LEVERAGE_UPGRADE.get(sym)
        if upgrade and len(_bars_up_to(histories, upgrade, as_of)) > _VOL_WINDOW:
            # 2x ETF → halve weight to maintain similar effective exposure;
            # the inverse-vol portfolio targeting already gave us a vol-
            # appropriate allocation in the unleveraged version.
            out[upgrade] = w * 0.5
        else:
            out[sym] = w
    return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@register
class Apex(Strategy):
    """Dual-momentum + vol-targeted leveraged risk parity with master switches.

    See module docstring for full design rationale and honest expectations.
    """

    bot_id = "apex"
    description = "Dual momentum + vol-targeting + trend/DD/VIX master switches"

    def target_weights(
        self,
        histories: dict[str, list[Bar]],
        as_of: date,
    ) -> dict[str, float]:
        # Master switch 1 — VIX panic
        vix_bars = _bars_up_to(histories, _VIX_SYMBOL, as_of)
        vix = vix_bars[-1].close if vix_bars else _VIX_CALM + 1
        if vix >= _VIX_PANIC:
            return _flight_to_quality(histories, as_of)

        # Master switch 2 — drawdown circuit breaker
        spy_bars = _bars_up_to(histories, "SPY", as_of)
        if _spy_drawdown(spy_bars) < _DD_CIRCUIT_BREAKER:
            return _defensive_mix(histories, as_of)

        # Master switch 3 — long-term trend filter
        if not _spy_uptrend(spy_bars):
            return _defensive_mix(histories, as_of)

        # Dual-momentum picks
        picks = _dual_momentum_picks(histories, as_of)
        if not picks:
            # No asset has positive 6m return → defensive
            return _defensive_mix(histories, as_of)

        # Vol-targeted inverse-vol weights on the picks
        weights = _vol_targeted_inverse_vol(
            histories, as_of, picks, target_vol=_TARGET_PORTFOLIO_VOL
        )
        if not weights:
            return _defensive_mix(histories, as_of)

        # Leverage gate: caution (VIX 25-35) keeps everything unleveraged.
        # Calm/normal regimes can upgrade SPY/TLT/GLD picks to SSO/TMF/UGL.
        if vix < _VIX_CAUTION:
            weights = _maybe_upgrade_to_leveraged(weights, histories, as_of)

        # Final cap on gross exposure (vol-target shouldn't exceed it but be safe)
        gross = sum(weights.values())
        if gross > _GROSS_LEVERAGE_CAP:
            scale = _GROSS_LEVERAGE_CAP / gross
            weights = {sym: w * scale for sym, w in weights.items()}

        return weights
