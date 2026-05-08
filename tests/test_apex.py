"""Tests for the Apex strategy: leveraged RP + VRP overlay + VIX kill switch."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.apex import (
    Apex,
    _GROSS_LEVERAGE_TARGET,
    _VRP_ALLOCATION,
)


def _bars(symbol: str, n: int, vol: float = 0.012, drift: float = 0.0003, seed: int = 0) -> list[Bar]:
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 100.0
    out: list[Bar] = []
    for i in range(n):
        ret = rng.gauss(drift, vol)
        price = max(price * (1 + ret), 0.01)
        out.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=1_000_000,
            )
        )
    return out


def _vix_bars(level: float, n: int = 120) -> list[Bar]:
    """Constant-VIX synthetic series so we can drive regime branches."""
    start = date(2020, 1, 6)
    return [
        Bar(symbol="^VIX", date=start + timedelta(days=i), open=level, high=level,
            low=level, close=level, volume=0)
        for i in range(n)
    ]


def _full_universe(vix_level: float, *, include_leveraged: bool = True, include_svxy: bool = True) -> dict[str, list[Bar]]:
    h: dict[str, list[Bar]] = {
        "SPY": _bars("SPY", 120, vol=0.012, seed=1),
        "TLT": _bars("TLT", 120, vol=0.008, seed=2),
        "GLD": _bars("GLD", 120, vol=0.010, seed=3),
        "SHY": _bars("SHY", 120, vol=0.001, seed=4),
        "^VIX": _vix_bars(vix_level),
    }
    if include_leveraged:
        h["SSO"] = _bars("SSO", 120, vol=0.024, seed=5)
        h["TMF"] = _bars("TMF", 120, vol=0.024, seed=6)
        h["UGL"] = _bars("UGL", 120, vol=0.020, seed=7)
    if include_svxy:
        h["SVXY"] = _bars("SVXY", 120, vol=0.030, seed=8)
    return h


def _as_of(h: dict[str, list[Bar]]) -> date:
    return max(b.date for bars in h.values() for b in bars)


# ---------------------------------------------------------------------------
# Regime branches
# ---------------------------------------------------------------------------


def test_panic_regime_holds_shy_only():
    h = _full_universe(vix_level=40.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert weights == {"SHY": _GROSS_LEVERAGE_TARGET}


def test_caution_regime_uses_unleveraged_sleeves():
    """VIX 25-35: deleverage to SPY/TLT/GLD even when SSO/TMF/UGL exist."""
    h = _full_universe(vix_level=28.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert "SSO" not in weights and "TMF" not in weights and "UGL" not in weights
    assert "SVXY" not in weights
    for sym in ("SPY", "TLT", "GLD"):
        assert sym in weights, f"{sym} missing from caution-regime weights"
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE_TARGET, abs=1e-6)


def test_normal_regime_prefers_leveraged_sleeves():
    """VIX 15-25: leveraged risk parity, NO SVXY overlay."""
    h = _full_universe(vix_level=18.0)
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights, f"{sym} missing from normal-regime weights"
    assert "SVXY" not in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE_TARGET, abs=1e-6)


def test_calm_regime_adds_svxy_overlay():
    """VIX < 15: leveraged RP scaled down 20% to make room for SVXY."""
    h = _full_universe(vix_level=12.0)
    weights = Apex().target_weights(h, _as_of(h))
    assert "SVXY" in weights
    assert weights["SVXY"] == pytest.approx(_VRP_ALLOCATION * _GROSS_LEVERAGE_TARGET, abs=1e-6)
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE_TARGET, abs=1e-6)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_falls_back_to_unleveraged_when_leveraged_missing():
    """SSO/TMF/UGL absent (e.g., 2008 stress) → use SPY/TLT/GLD."""
    h = _full_universe(vix_level=18.0, include_leveraged=False)
    weights = Apex().target_weights(h, _as_of(h))
    for sym in ("SPY", "TLT", "GLD"):
        assert sym in weights, f"{sym} missing in fallback weights"
    assert "SSO" not in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE_TARGET, abs=1e-6)


def test_calm_without_svxy_returns_pure_leveraged_rp():
    """VIX < 15 but SVXY history absent → no overlay, just leveraged RP."""
    h = _full_universe(vix_level=10.0, include_svxy=False)
    weights = Apex().target_weights(h, _as_of(h))
    assert "SVXY" not in weights
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE_TARGET, abs=1e-6)


def test_panic_without_shy_returns_empty():
    """No defensive instrument → empty weights (engine holds cash)."""
    h = _full_universe(vix_level=40.0)
    del h["SHY"]
    weights = Apex().target_weights(h, _as_of(h))
    assert weights == {}


def test_no_vix_data_assumes_normal():
    """Missing ^VIX → default to NORMAL regime (no kill switch)."""
    h = _full_universe(vix_level=18.0)
    del h["^VIX"]
    weights = Apex().target_weights(h, _as_of(h))
    # NORMAL: leveraged RP, no SVXY
    for sym in ("SSO", "TMF", "UGL"):
        assert sym in weights
    assert "SVXY" not in weights


# ---------------------------------------------------------------------------
# Inverse-vol property
# ---------------------------------------------------------------------------


def test_higher_vol_leg_gets_lower_weight():
    """Inverse-vol weighting: more volatile leg → smaller allocation."""
    h: dict[str, list[Bar]] = {
        "^VIX": _vix_bars(18.0),
        "SSO": _bars("SSO", 120, vol=0.040, seed=1),  # very volatile
        "TMF": _bars("TMF", 120, vol=0.012, seed=2),
        "UGL": _bars("UGL", 120, vol=0.020, seed=3),
        "SPY": _bars("SPY", 120, seed=4),
        "TLT": _bars("TLT", 120, seed=5),
        "GLD": _bars("GLD", 120, seed=6),
        "SHY": _bars("SHY", 120, seed=7),
    }
    weights = Apex().target_weights(h, _as_of(h))
    # TMF (lowest vol) should have the highest weight, SSO (highest) the lowest
    assert weights["TMF"] > weights["UGL"] > weights["SSO"]
