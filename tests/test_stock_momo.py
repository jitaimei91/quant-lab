"""Tests for stock_momo (top-decile S&P 500 momentum bot)."""
from __future__ import annotations

from datetime import date, timedelta
from unittest import mock

import pytest

from quant_lab.types import Bar
from quant_lab.strategies.stock_momo import (
    StockMomo,
    _GROSS_LEVERAGE,
    _MAX_PICKS,
    _MIN_PICKS,
)


def _bars(symbol: str, n: int, daily_return: float, vol: float = 0.005, seed: int = 0) -> list[Bar]:
    import random
    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 100.0
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
# Universe behaviour
# ---------------------------------------------------------------------------


def test_picks_only_from_stock_universe():
    """The bot must NOT pick tickers outside its S&P 500 universe."""
    fake_universe = frozenset({"AAA", "BBB", "CCC"})
    h = {
        "AAA": _bars("AAA", 280, daily_return=0.001, seed=1),
        "BBB": _bars("BBB", 280, daily_return=0.002, seed=2),
        "CCC": _bars("CCC", 280, daily_return=0.0005, seed=3),
        # Foreign ticker — not in our universe, must be ignored even if it has
        # the highest momentum.
        "XYZ": _bars("XYZ", 280, daily_return=0.005, seed=4),
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert "XYZ" not in weights
    for sym in weights:
        assert sym in fake_universe


def test_negative_momentum_excluded():
    """Tickers with non-positive 6mo return must NOT be picked."""
    fake_universe = frozenset({"WIN1", "WIN2", "LOSE"})
    h = {
        "WIN1": _bars("WIN1", 280, daily_return=0.002, seed=1),
        "WIN2": _bars("WIN2", 280, daily_return=0.001, seed=2),
        "LOSE": _bars("LOSE", 280, daily_return=-0.002, seed=3),
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert "LOSE" not in weights


def test_recent_ipo_excluded_via_min_history():
    """Names with fewer than 252 bars must be excluded — no recent-IPO picks."""
    fake_universe = frozenset({"OLD", "NEW"})
    h = {
        "OLD": _bars("OLD", 280, daily_return=0.001, seed=1),
        "NEW": _bars("NEW", 100, daily_return=0.005, seed=2),  # only 100 days history
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert "NEW" not in weights
    assert "OLD" in weights


def test_no_positive_momentum_returns_empty():
    """When no name has positive 6mo return, return empty dict (cash).
    Use strong negative drift + tiny vol so the random walk can't overshoot positive."""
    fake_universe = frozenset({"L1", "L2"})
    h = {
        "L1": _bars("L1", 280, daily_return=-0.005, vol=0.001, seed=1),
        "L2": _bars("L2", 280, daily_return=-0.008, vol=0.001, seed=2),
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert weights == {}


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_equal_weighted_with_5pct_cash_buffer():
    """All picks must have identical weight summing to 95%."""
    fake_universe = frozenset({f"S{i}" for i in range(20)})
    h = {
        sym: _bars(sym, 280, daily_return=0.001 + 0.0001 * i, seed=i)
        for i, sym in enumerate(sorted(fake_universe))
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    # All weights identical
    weight_values = list(weights.values())
    assert len(set(round(v, 9) for v in weight_values)) == 1
    # Total ≈ 95%
    assert sum(weights.values()) == pytest.approx(_GROSS_LEVERAGE, abs=1e-6)


def test_holds_at_least_min_picks_when_universe_small():
    """With 20 candidates, top-decile = 2; but _MIN_PICKS forces ≥10."""
    fake_universe = frozenset({f"S{i}" for i in range(20)})
    h = {
        sym: _bars(sym, 280, daily_return=0.001 + 0.0001 * i, seed=i)
        for i, sym in enumerate(sorted(fake_universe))
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert len(weights) >= _MIN_PICKS or len(weights) == 20
    assert len(weights) <= 20  # can't exceed universe size


def test_caps_at_max_picks_when_universe_large():
    """With 1000 candidates, top-decile = 100; but _MAX_PICKS caps at 50."""
    fake_universe = frozenset({f"S{i}" for i in range(600)})
    h = {
        sym: _bars(sym, 280, daily_return=0.001 + 0.0001 * (i % 100), seed=i % 100)
        for i, sym in enumerate(sorted(fake_universe))
    }
    with mock.patch("quant_lab.strategies.stock_momo.STOCK_UNIVERSE", fake_universe):
        weights = StockMomo().target_weights(h, _as_of(h))
    assert len(weights) <= _MAX_PICKS


# ---------------------------------------------------------------------------
# Universe loading helper
# ---------------------------------------------------------------------------


def test_universe_loads_from_file(tmp_path):
    """load_universe correctly skips comments and blank lines."""
    from quant_lab.strategies._universe import load_universe

    f = tmp_path / "u.txt"
    f.write_text("# a comment\n\nAAPL\n# another\nMSFT\n  GOOGL  \n")
    out = load_universe(f)
    assert out == frozenset({"AAPL", "MSFT", "GOOGL"})


def test_universe_returns_empty_for_missing_file(tmp_path):
    from quant_lab.strategies._universe import load_universe
    out = load_universe(tmp_path / "nonexistent.txt")
    assert out == frozenset()
