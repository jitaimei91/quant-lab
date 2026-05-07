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

import math
import statistics
from datetime import date, timedelta
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Bar
from .hmm_regime import HMMState, forward_backward, load_hmm


TRADING_DAYS_PER_YEAR = 252

_VIX_CAUTION = 35.0
_VIX_PANIC = 50.0

_DD_PAUSE_THRESHOLD = -0.25   # 25% drawdown triggers pause
_SHARPE_PAUSE_THRESHOLD = -1.0


_DEFAULT_HMM_PATH = Path(__file__).resolve().parents[3] / "state" / "hmm_state.json"


def regime_state(
    histories: dict[str, list[Bar]],
    hmm_state_path: Path | None = None,
) -> dict[str, Any]:
    """Compute current regime from VIX history.

    Returns
    -------
    dict with keys:
        vix              — latest VIX close (0.0 if unavailable)
        regime           — "NORMAL" | "CAUTION" | "PANIC"
        halt_new_entries — bool
        liquidate_all    — bool
        hmm              — HMM regime classification dict or None
    """
    resolved_hmm_path = hmm_state_path if hmm_state_path is not None else _DEFAULT_HMM_PATH

    vix_bars = histories.get("^VIX", [])
    if not vix_bars:
        # No VIX data → assume NORMAL (safe default)
        return {
            "vix": 0.0,
            "regime": "NORMAL",
            "halt_new_entries": False,
            "liquidate_all": False,
            "hmm": None,
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

    # HMM classification: only when a trained model exists
    hmm_result: dict[str, Any] | None = None
    if resolved_hmm_path.exists():
        try:
            hmm_result = hmm_regime_classify(histories, resolved_hmm_path)
        except Exception:
            hmm_result = None

    return {
        "vix": vix,
        "regime": regime,
        "halt_new_entries": halt_new_entries,
        "liquidate_all": liquidate_all,
        "hmm": hmm_result,
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


# ---------------------------------------------------------------------------
# HMM regime classification
# ---------------------------------------------------------------------------

_HMM_REGIME_NAMES = ["risk-on", "chop", "risk-off", "crisis"]


def _hmm_observations(histories: dict[str, list[Bar]], lookback: int = 60) -> np.ndarray:
    """Build the daily observation matrix for the HMM.

    Features (5 total):
        0: VIX level (^VIX close)
        1: VIX 1-day log return
        2: SPY 20-day return
        3: SPY 20-day realized vol (annualized)
        4: TLT/SHY return differential (term spread proxy) -- 0 if unavailable

    Returns
    -------
    (lookback, 5)  -- zeros for missing series
    """
    result = np.zeros((lookback, 5))

    vix_bars = histories.get("^VIX", [])
    spy_bars = histories.get("SPY", [])
    tlt_bars = histories.get("TLT", [])
    shy_bars = histories.get("SHY", [])

    # Feature 0 & 1: VIX level and 1-day log return
    if vix_bars:
        vix_slice = vix_bars[-(lookback + 1):]
        for i in range(lookback):
            vix_idx = i - lookback + len(vix_slice) - 1
            if 0 <= vix_idx < len(vix_slice):
                result[i, 0] = vix_slice[vix_idx].close
                if vix_idx > 0 and vix_slice[vix_idx - 1].close > 0:
                    result[i, 1] = math.log(vix_slice[vix_idx].close / vix_slice[vix_idx - 1].close)

    # Feature 2 & 3: SPY 20-day return and realized vol
    if spy_bars:
        spy_slice = spy_bars[-(lookback + 21):]
        for i in range(lookback):
            end_idx = i - lookback + len(spy_slice) - 1
            start_idx = end_idx - 20
            if 0 <= start_idx and end_idx < len(spy_slice):
                end_price = spy_slice[end_idx].close
                start_price = spy_slice[start_idx].close
                if start_price > 0:
                    result[i, 2] = end_price / start_price - 1.0
                window_bars = spy_slice[start_idx:end_idx + 1]
                if len(window_bars) >= 2:
                    daily_rets = [
                        math.log(window_bars[j].close / window_bars[j - 1].close)
                        for j in range(1, len(window_bars))
                        if window_bars[j - 1].close > 0
                    ]
                    if len(daily_rets) >= 2:
                        std_ret = statistics.stdev(daily_rets)
                        result[i, 3] = std_ret * math.sqrt(252)

    # Feature 4: TLT/SHY return differential (term spread proxy)
    if tlt_bars and shy_bars:
        tlt_slice = tlt_bars[-(lookback + 1):]
        shy_slice = shy_bars[-(lookback + 1):]
        for i in range(lookback):
            tlt_idx = i - lookback + len(tlt_slice) - 1
            shy_idx = i - lookback + len(shy_slice) - 1
            if 0 <= tlt_idx < len(tlt_slice) and tlt_idx > 0:
                if tlt_slice[tlt_idx - 1].close > 0:
                    tlt_ret = math.log(tlt_slice[tlt_idx].close / tlt_slice[tlt_idx - 1].close)
                else:
                    tlt_ret = 0.0
            else:
                tlt_ret = 0.0
            if 0 <= shy_idx < len(shy_slice) and shy_idx > 0:
                if shy_slice[shy_idx - 1].close > 0:
                    shy_ret = math.log(shy_slice[shy_idx].close / shy_slice[shy_idx - 1].close)
                else:
                    shy_ret = 0.0
            else:
                shy_ret = 0.0
            result[i, 4] = tlt_ret - shy_ret

    return result


def _map_states_to_regimes(hmm: HMMState) -> dict[int, str]:
    """Map HMM state indices to regime names by sorting on mean VIX (feature 0).

    Lowest mean VIX -> 'risk-on', then 'chop', 'risk-off', highest -> 'crisis'.
    """
    vix_means = [(hmm.means[k, 0], k) for k in range(hmm.n_states)]
    vix_means.sort()
    names = _HMM_REGIME_NAMES[: hmm.n_states]
    return {state_idx: names[rank] for rank, (_, state_idx) in enumerate(vix_means)}


def hmm_regime_classify(histories: dict[str, list[Bar]], hmm_state_path: Path) -> dict[str, Any]:
    """Classify current market regime using a trained HMM.

    Parameters
    ----------
    histories : symbol -> list of Bar
    hmm_state_path : path to the saved HMMState JSON

    Returns
    -------
    dict with keys:
        regime_id          — int (0-3)
        regime_name        — str ('risk-on' / 'chop' / 'risk-off' / 'crisis')
        regime_probs       — dict[str, float]
        regime_confidence  — float (max posterior)
    """
    hmm = load_hmm(hmm_state_path)
    if hmm is None:
        return {
            "regime_id": 0,
            "regime_name": "risk-on",
            "regime_probs": {name: 0.25 for name in _HMM_REGIME_NAMES},
            "regime_confidence": 0.25,
        }

    obs = _hmm_observations(histories, lookback=60)
    posteriors = forward_backward(obs, hmm)

    latest = posteriors[-1]  # (n_states,)
    regime_id = int(np.argmax(latest))
    confidence = float(latest[regime_id])

    state_to_name = _map_states_to_regimes(hmm)
    regime_name = state_to_name[regime_id]

    regime_probs: dict[str, float] = {name: 0.0 for name in _HMM_REGIME_NAMES[: hmm.n_states]}
    for k in range(hmm.n_states):
        name = state_to_name[k]
        regime_probs[name] = float(latest[k])

    return {
        "regime_id": regime_id,
        "regime_name": regime_name,
        "regime_probs": regime_probs,
        "regime_confidence": confidence,
    }
