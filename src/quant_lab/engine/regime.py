"""Regime kill-switch: VIX thresholds + per-bot drawdown halts.

Regime states:
    NORMAL   — VIX < 35
    CAUTION  — 35 <= VIX < 50 (halt new entries, keep existing positions)
    PANIC    — VIX >= 50 (liquidate all positions)

Per-bot pauses fire on:
    - 30-day trailing drawdown > 25%
    - 60-day rolling Sharpe < -1.0
"""
from __future__ import annotations

import statistics
from datetime import date, timedelta
from math import sqrt
from typing import Any

from ..types import Bar


TRADING_DAYS_PER_YEAR = 252

_VIX_CAUTION = 35.0
_VIX_PANIC = 50.0

_DD_PAUSE_THRESHOLD = -0.25   # 25% drawdown triggers pause
_SHARPE_PAUSE_THRESHOLD = -1.0


def regime_state(histories: dict[str, list[Bar]]) -> dict[str, Any]:
    """Compute current regime from VIX history.

    Returns
    -------
    dict with keys:
        vix           — latest VIX close (0.0 if unavailable)
        regime        — "NORMAL" | "CAUTION" | "PANIC"
        halt_new_entries — bool
        liquidate_all    — bool
    """
    vix_bars = histories.get("^VIX", [])
    if not vix_bars:
        # No VIX data → assume NORMAL (safe default)
        return {
            "vix": 0.0,
            "regime": "NORMAL",
            "halt_new_entries": False,
            "liquidate_all": False,
        }

    vix = vix_bars[-1].close

    if vix >= _VIX_PANIC:
        regime = "PANIC"
        halt_new_entries = True
        liquidate_all = True
    elif vix >= _VIX_CAUTION:
        regime = "CAUTION"
        halt_new_entries = True
        liquidate_all = False
    else:
        regime = "NORMAL"
        halt_new_entries = False
        liquidate_all = False

    return {
        "vix": vix,
        "regime": regime,
        "halt_new_entries": halt_new_entries,
        "liquidate_all": liquidate_all,
    }


def per_bot_drawdown(
    nav_series: list[tuple[date, float]],
    window_days: int = 30,
) -> float:
    """Compute trailing-window max drawdown.

    Returns a non-positive float (e.g. -0.25 for 25% drawdown).
    Returns 0.0 if insufficient data.
    """
    if not nav_series:
        return 0.0

    cutoff = nav_series[-1][0] - timedelta(days=window_days)
    window = [nav for dt, nav in nav_series if dt >= cutoff]
    if len(window) < 2:
        return 0.0

    peak = window[0]
    worst = 0.0
    for nav in window:
        peak = max(peak, nav)
        if peak > 0:
            dd = (nav - peak) / peak
            worst = min(worst, dd)
    return worst


def should_pause_bot(
    bot_id: str,
    nav_series: list[tuple[date, float]],
    sharpe_window_days: int = 60,
) -> tuple[bool, str]:
    """Check whether a bot should be paused.

    Pauses when:
    - 30-day trailing drawdown < -25%
    - 60-day rolling Sharpe < -1.0

    Returns
    -------
    (should_pause: bool, reason: str)
    """
    if not nav_series:
        return False, ""

    # Drawdown check (30-day window)
    dd = per_bot_drawdown(nav_series, window_days=30)
    if dd < _DD_PAUSE_THRESHOLD:
        return True, f"30-day drawdown {dd:.1%} < {_DD_PAUSE_THRESHOLD:.0%} threshold"

    # Sharpe check (60-day window)
    cutoff = nav_series[-1][0] - timedelta(days=sharpe_window_days)
    window_navs = [nav for dt, nav in nav_series if dt >= cutoff]
    if len(window_navs) >= 10:
        rets = [window_navs[i] / window_navs[i - 1] - 1.0 for i in range(1, len(window_navs))]
        if len(rets) >= 2:
            mean_r = statistics.mean(rets)
            std_r = statistics.stdev(rets)
            if std_r > 0:
                rolling_sharpe = (mean_r / std_r) * sqrt(TRADING_DAYS_PER_YEAR)
                if rolling_sharpe < _SHARPE_PAUSE_THRESHOLD:
                    return True, f"60-day Sharpe {rolling_sharpe:.2f} < {_SHARPE_PAUSE_THRESHOLD} threshold"

    return False, ""
