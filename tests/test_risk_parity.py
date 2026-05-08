"""Tests for the RiskParity sleeve strategy."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.risk_parity import RiskParity, _LEGS


def _bars(symbol: str, n: int, drift: float = 0.0003, vol: float = 0.012, seed: int = 0) -> list[Bar]:
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


def _histories(spy_vol: float = 0.012, tlt_vol: float = 0.008, gld_vol: float = 0.010) -> dict[str, list[Bar]]:
    return {
        "SPY": _bars("SPY", n=120, vol=spy_vol, seed=1),
        "TLT": _bars("TLT", n=120, vol=tlt_vol, seed=2),
        "GLD": _bars("GLD", n=120, vol=gld_vol, seed=3),
    }


def test_risk_parity_allocates_to_all_three_legs():
    bot = RiskParity()
    histories = _histories()
    as_of = date(2020, 1, 6) + timedelta(days=119)
    weights = bot.target_weights(histories, as_of)
    for leg in _LEGS:
        assert leg in weights, f"missing {leg} in weights"
    assert sum(weights.values()) == pytest.approx(0.95, abs=1e-6)


def test_risk_parity_higher_weight_to_lower_vol():
    """Inverse-vol weighting → lower-vol leg gets higher weight."""
    bot = RiskParity()
    # TLT very low vol, SPY high vol, GLD medium
    histories = _histories(spy_vol=0.020, tlt_vol=0.005, gld_vol=0.012)
    as_of = date(2020, 1, 6) + timedelta(days=119)
    weights = bot.target_weights(histories, as_of)
    assert weights["TLT"] > weights["GLD"] > weights["SPY"]


def test_risk_parity_skips_legs_with_insufficient_data():
    """A leg with too few bars is dropped; others are still weighted."""
    bot = RiskParity()
    histories = _histories()
    histories["TLT"] = _bars("TLT", n=10, seed=99)  # well below the 60-day window
    as_of = date(2020, 1, 6) + timedelta(days=119)
    weights = bot.target_weights(histories, as_of)
    assert "TLT" not in weights
    assert "SPY" in weights and "GLD" in weights
    assert sum(weights.values()) == pytest.approx(0.95, abs=1e-6)


def test_risk_parity_returns_empty_when_no_data():
    bot = RiskParity()
    as_of = date(2020, 1, 6)
    assert bot.target_weights({}, as_of) == {}
