"""Tests for sector_momentum, credit_carry, cross_asset_trend bots."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.sector_momentum import SectorMomentum, _SECTORS, _DEFENSIVE_FALLBACK
from quant_lab.strategies.credit_carry import CreditCarry
from quant_lab.strategies.cross_asset_trend import CrossAssetTrend, _LEGS as _XAT_LEGS, _FALLBACK as _XAT_FALLBACK


def _bars(symbol: str, n: int, daily_return: float, vol: float = 0.005, *, base: float = 100.0, seed: int = 0) -> list[Bar]:
    import random
    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = base
    out: list[Bar] = []
    for i in range(n):
        ret = daily_return + rng.gauss(0.0, vol)
        price = max(price * (1 + ret), 0.01)
        out.append(
            Bar(symbol=symbol, date=start + timedelta(days=i),
                open=price, high=price * 1.005, low=price * 0.995,
                close=price, volume=1_000_000)
        )
    return out


def _as_of(h: dict[str, list[Bar]]) -> date:
    return max(b.date for bars in h.values() for b in bars)


# ---------------------------------------------------------------------------
# SectorMomentum
# ---------------------------------------------------------------------------


def test_sector_momentum_picks_top_3_with_positive_return():
    h = {sym: _bars(sym, 250, daily_return=0.001 * (i + 1), vol=0.005, seed=i)
         for i, sym in enumerate(_SECTORS[:6])}
    h["IEF"] = _bars("IEF", 250, daily_return=0.0001, vol=0.001, seed=99)
    weights = SectorMomentum().target_weights(h, _as_of(h))
    assert 1 <= len(weights) <= 3
    # All chosen syms must come from the sector universe
    for sym in weights:
        assert sym in _SECTORS or sym == _DEFENSIVE_FALLBACK


def test_sector_momentum_falls_back_to_ief_when_all_negative():
    h = {sym: _bars(sym, 250, daily_return=-0.001, vol=0.005, seed=i)
         for i, sym in enumerate(_SECTORS)}
    h["IEF"] = _bars("IEF", 250, daily_return=0.0001, vol=0.001, seed=99)
    weights = SectorMomentum().target_weights(h, _as_of(h))
    assert weights == {"IEF": pytest.approx(0.95)}


def test_sector_momentum_returns_empty_when_universe_missing():
    weights = SectorMomentum().target_weights({}, date(2026, 5, 9))
    assert weights == {}


# ---------------------------------------------------------------------------
# CreditCarry
# ---------------------------------------------------------------------------


def test_credit_carry_holds_hyg_lqd_in_uptrend():
    h = {
        "SPY": _bars("SPY", 250, daily_return=0.001, vol=0.003, seed=1),
        "HYG": _bars("HYG", 250, daily_return=0.0003, vol=0.002, seed=2),
        "LQD": _bars("LQD", 250, daily_return=0.0002, vol=0.002, seed=3),
        "IEF": _bars("IEF", 250, daily_return=0.0001, vol=0.001, seed=4),
    }
    weights = CreditCarry().target_weights(h, _as_of(h))
    assert "HYG" in weights and "LQD" in weights
    assert "IEF" not in weights


def test_credit_carry_holds_ief_in_downtrend():
    h = {
        # Strong negative drift, tiny noise → SPY < 200d MA reliably
        "SPY": _bars("SPY", 250, daily_return=-0.003, vol=0.001, base=200.0, seed=1),
        "HYG": _bars("HYG", 250, daily_return=0.0003, vol=0.002, seed=2),
        "LQD": _bars("LQD", 250, daily_return=0.0002, vol=0.002, seed=3),
        "IEF": _bars("IEF", 250, daily_return=0.0001, vol=0.001, seed=4),
    }
    weights = CreditCarry().target_weights(h, _as_of(h))
    assert weights == {"IEF": pytest.approx(0.95)}


def test_credit_carry_falls_through_to_ief_when_credit_legs_missing():
    h = {
        "SPY": _bars("SPY", 250, daily_return=0.001, vol=0.003, seed=1),
        "IEF": _bars("IEF", 250, daily_return=0.0001, vol=0.001, seed=2),
    }
    weights = CreditCarry().target_weights(h, _as_of(h))
    assert weights == {"IEF": pytest.approx(0.95)}


# ---------------------------------------------------------------------------
# CrossAssetTrend
# ---------------------------------------------------------------------------


def test_cross_asset_trend_picks_only_positive_momentum_legs():
    h = {
        "SPY": _bars("SPY", 280, daily_return=0.001, vol=0.005, seed=1),
        "EFA": _bars("EFA", 280, daily_return=-0.001, vol=0.005, seed=2),
        "EEM": _bars("EEM", 280, daily_return=0.0008, vol=0.008, seed=3),
        "TLT": _bars("TLT", 280, daily_return=-0.0005, vol=0.005, seed=4),
        "GLD": _bars("GLD", 280, daily_return=0.0005, vol=0.006, seed=5),
        "USO": _bars("USO", 280, daily_return=-0.001, vol=0.012, seed=6),
        "IEF": _bars("IEF", 280, daily_return=0.0001, vol=0.001, seed=7),
    }
    weights = CrossAssetTrend().target_weights(h, _as_of(h))
    # Positive-momo legs: SPY, EEM, GLD; negative: EFA, TLT, USO
    for sym in ("EFA", "TLT", "USO"):
        assert sym not in weights
    assert any(sym in weights for sym in ("SPY", "EEM", "GLD"))


def test_cross_asset_trend_falls_back_when_all_negative():
    h = {sym: _bars(sym, 280, daily_return=-0.001, vol=0.005, seed=i)
         for i, sym in enumerate(_XAT_LEGS)}
    h["IEF"] = _bars("IEF", 280, daily_return=0.0001, vol=0.001, seed=99)
    weights = CrossAssetTrend().target_weights(h, _as_of(h))
    assert weights == {_XAT_FALLBACK: pytest.approx(0.95)}
