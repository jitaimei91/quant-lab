"""Tests for Apex v2: trend-filtered leveraged RP + DD circuit breaker."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.apex import (
    Apex,
    _GROSS_LEVERAGE,
    _DD_CIRCUIT_BREAKER,
)


# ---------------------------------------------------------------------------
# Bar generators
# ---------------------------------------------------------------------------


def _trending_up_bars(symbol: str, n: int = 250, daily_return: float = 0.001, vol: float = 0.005, seed: int = 0) -> list[Bar]:
    """Bars that trend up cleanly so SPY > 200d MA."""
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 100.0
    out: list[Bar] = []
    for i in range(n):
        ret = daily_return + rng.gauss(0.0, vol)
        price = max(price * (1 + ret), 0.01)
        out.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price, high=price * 1.005, low=price * 0.995,
                close=price, volume=1_000_000,
            )
        )
    return out


def _trending_down_bars(symbol: str, n: int = 250, seed: int = 0) -> list[Bar]:
    """Bars that trend down so SPY < 200d MA."""
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 200.0  # start high, drift down
    out: list[Bar] = []
    for i in range(n):
        ret = -0.001 + rng.gauss(0.0, 0.005)
        price = max(price * (1 + ret), 0.01)
        out.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price, high=price * 1.005, low=price * 0.995,
                close=price, volume=1_000_000,
            )
        )
    return out


def _crash_bars(symbol: str, n_pre: int = 200, crash_pct: float = -0.20, seed: int = 0) -> list[Bar]:
    """Steady uptrend followed by a sharp 20% drawdown over the last 30 bars."""
    pre = _trending_up_bars(symbol, n=n_pre, daily_return=0.001, seed=seed)
    last = pre[-1]
    crash_per_day = crash_pct / 30
    out = list(pre)
    price = last.close
    for i in range(30):
        price *= 1 + crash_per_day
        out.append(
            Bar(
                symbol=symbol,
                date=last.date + timedelta(days=i + 1),
                open=price, high=price, low=price, close=price, volume=1_000_000,
            )
        )
    return out


def _vix_bars(level: float, n: int = 250) -> list[Bar]:
    start = date(2020, 1, 6)
    return [
        Bar(symbol="^VIX", date=start + timedelta(days=i),
            open=level, high=level, low=level, close=level, volume=0)
        for i in range(n)
    ]


def _full_universe(spy_pattern: str, vix_level: float, *, include_leveraged: bool = True) -> dict[str, list[Bar]]:
    """spy_pattern: 'up' | 'down' | 'crash'"""
    if spy_pattern == "up":
        spy = _trending_up_bars("SPY", n=250, daily_return=0.001, seed=1)
    elif spy_pattern == "down":
        spy = _trending_down_bars("SPY", n=250, seed=1)
    else:  # crash
        spy = _crash_bars("SPY", n_pre=200, crash_pct=-0.20, seed=1)

    h: dict[str, list[Bar]] = {
        "SPY": spy,
        "TLT": _trending_up_bars("TLT", n=250, daily_return=0.0003, seed=2),
        "GLD": _trending_up_bars("GLD", n=250, daily_return=0.0004, seed=3),
        "IEF": _trending_up_bars("IEF", n=250, daily_return=0.0001, vol=0.002, seed=4),
        "SHY": _trending_up_bars("SHY", n=250, daily_return=0.00005, vol=0.0005, seed=5),
        "^VIX": _vix_bars(vix_level, n=250),
    }
    if include_leveraged:
        h["SSO"] = _trending_up_bars("SSO", n=250, daily_return=0.002, vol=0.01, seed=6)
        h["TMF"] = _trending_up_bars("TMF", n=250, daily_return=0.0009, vol=0.012, seed=7)
        h["UGL"] = _trending_up_bars("UGL", n=250, daily_return=0.0008, vol=0.010, seed=8)
    return h


def _as_of(h: dict[str, list[Bar]]) -> date:
    return max(b.date for bars in h.values() for b in bars)


# ---------------------------------------------------------------------------
# Trend filter: uptrend + low VIX → leveraged RP
# ---------------------------------------------------------------------------


def test_uptrend_normal_vix_uses_leveraged_rp():
    h = _full_universe("up", vix_level=18.0)
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE, abs=1e-6)


def test_uptrend_calm_vix_still_uses_leveraged_rp_no_svxy():
    """v2 does NOT add SVXY in calm regime — that bot is dead."""
    h = _full_universe("up", vix_level=10.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert "SVXY" not in weights
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights


def test_uptrend_caution_uses_unleveraged_rp():
    h = _full_universe("up", vix_level=28.0)
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SPY", "TLT", "GLD"):
        assert sym in weights
    for sym in ("SSO", "TMF", "UGL"):
        assert sym not in weights


# ---------------------------------------------------------------------------
# Defensive mix on downtrend / drawdown / panic
# ---------------------------------------------------------------------------


def test_downtrend_goes_defensive_even_with_low_vix():
    """SPY < 200d MA → defensive 50/50 IEF/SHY regardless of low VIX."""
    h = _full_universe("down", vix_level=10.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert "IEF" in weights and "SHY" in weights
    # No leveraged or risky assets
    for sym in ("SSO", "TMF", "UGL", "SPY"):
        assert sym not in weights


def test_drawdown_circuit_breaker_overrides_trend():
    """SPY 60d DD > 15% triggers defensive even if 200d MA is still bullish."""
    h = _full_universe("crash", vix_level=18.0)
    spy_bars = h["SPY"]
    # Sanity check on the synthetic data: drawdown should exceed the threshold
    peak = max(b.close for b in spy_bars[-60:])
    dd = spy_bars[-1].close / peak - 1.0
    assert dd < _DD_CIRCUIT_BREAKER, f"test data only achieves {dd:.2%} DD"
    weights = Apex().target_weights(h, _as_of(h))
    assert "IEF" in weights and "SHY" in weights
    for sym in ("SSO", "TMF", "UGL"):
        assert sym not in weights


def test_panic_vix_goes_to_ief_flight_to_quality():
    """VIX >= 35 → 100% IEF (NOT SHY — capture flight-to-quality bond rally)."""
    h = _full_universe("up", vix_level=40.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert weights == {"IEF": _GROSS_LEVERAGE}


def test_panic_vix_overrides_drawdown_and_trend():
    """Even in a crash with downtrend, panic regime still routes to IEF."""
    h = _full_universe("crash", vix_level=40.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert weights == {"IEF": _GROSS_LEVERAGE}


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_falls_back_to_unleveraged_when_leveraged_missing():
    """SSO/TMF/UGL absent → use SPY/TLT/GLD inverse-vol."""
    h = _full_universe("up", vix_level=18.0, include_leveraged=False)
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SPY", "TLT", "GLD"):
        assert sym in weights
    assert "SSO" not in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE, abs=1e-6)


def test_panic_falls_back_when_ief_missing():
    """Panic → IEF, but if IEF missing fall through to TLT, then SHY."""
    h = _full_universe("up", vix_level=40.0)
    del h["IEF"]
    weights = Apex().target_weights(h, _as_of(h))
    # With IEF missing, falls through to TLT
    assert weights == {"TLT": _GROSS_LEVERAGE}


def test_no_vix_data_assumes_uptrend_normal():
    h = _full_universe("up", vix_level=18.0)
    del h["^VIX"]
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights


def test_short_spy_history_assumes_uptrend():
    """When SPY history is too short for 200d MA, assume bullish (no false defensive)."""
    h = _full_universe("up", vix_level=18.0)
    h["SPY"] = h["SPY"][:50]  # only 50 days, not enough for 200d MA
    weights = Apex().target_weights(h, _as_of(h))
    # Should still allocate to leveraged sleeves rather than incorrectly going defensive
    assert "SSO" in weights or "TMF" in weights or "UGL" in weights


# ---------------------------------------------------------------------------
# Inverse-vol property (regression check)
# ---------------------------------------------------------------------------


def test_higher_vol_leg_gets_lower_weight_in_leveraged_rp():
    """When in leveraged RP regime, the higher-vol leg gets the smaller weight."""
    base = _full_universe("up", vix_level=18.0)
    # Override SSO with very high vol
    import random
    rng = random.Random(99)
    start = date(2020, 1, 6)
    price = 100.0
    high_vol_bars: list[Bar] = []
    for i in range(250):
        ret = 0.001 + rng.gauss(0.0, 0.04)  # 4% daily vol — very high
        price = max(price * (1 + ret), 0.01)
        high_vol_bars.append(
            Bar(symbol="SSO", date=start + timedelta(days=i),
                open=price, high=price, low=price, close=price, volume=1_000_000)
        )
    base["SSO"] = high_vol_bars
    weights = Apex().target_weights(base, _as_of(base))
    # SSO (high vol) should have LESS weight than TMF or UGL
    assert weights["SSO"] < weights["TMF"]
    assert weights["SSO"] < weights["UGL"]
