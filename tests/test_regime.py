"""Tests for engine/regime.py — VIX kill-switch + per-bot drawdown halts."""
from __future__ import annotations

from datetime import date, timedelta

from quant_lab.engine.regime import per_bot_drawdown, regime_state, should_pause_bot
from quant_lab.engine.paper import rebalance
from quant_lab.types import Bar, Portfolio, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vix_bar(close: float) -> Bar:
    return Bar(symbol="^VIX", date=date(2026, 5, 7), open=close, high=close, low=close, close=close, volume=0)


def _nav_series(values: list[float], start: date | None = None) -> list[tuple[date, float]]:
    if start is None:
        start = date(2026, 1, 2)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


# ---------------------------------------------------------------------------
# Regime state tests
# ---------------------------------------------------------------------------

def test_vix_normal():
    """VIX = 25 → NORMAL regime."""
    state = regime_state({"^VIX": [_vix_bar(25.0)]})
    assert state["regime"] == "NORMAL"
    assert state["halt_new_entries"] is False
    assert state["liquidate_all"] is False
    assert state["vix"] == 25.0


def test_vix_caution():
    """VIX = 40 → CAUTION, halt_new_entries=True."""
    state = regime_state({"^VIX": [_vix_bar(40.0)]})
    assert state["regime"] == "CAUTION"
    assert state["halt_new_entries"] is True
    assert state["liquidate_all"] is False


def test_vix_panic():
    """VIX = 55 → PANIC, liquidate_all=True."""
    state = regime_state({"^VIX": [_vix_bar(55.0)]})
    assert state["regime"] == "PANIC"
    assert state["halt_new_entries"] is True
    assert state["liquidate_all"] is True


def test_no_vix_data_normal():
    """Missing ^VIX → safe default NORMAL."""
    state = regime_state({})
    assert state["regime"] == "NORMAL"
    assert state["vix"] == 0.0


def test_vix_boundary_exactly_35():
    """VIX == 35.0 → CAUTION (>= threshold)."""
    state = regime_state({"^VIX": [_vix_bar(35.0)]})
    assert state["regime"] == "CAUTION"


def test_vix_boundary_exactly_50():
    """VIX == 50.0 → PANIC (>= threshold)."""
    state = regime_state({"^VIX": [_vix_bar(50.0)]})
    assert state["regime"] == "PANIC"


# ---------------------------------------------------------------------------
# Per-bot drawdown tests
# ---------------------------------------------------------------------------

def test_per_bot_drawdown_no_data():
    """Empty series → 0.0."""
    assert per_bot_drawdown([]) == 0.0


def test_per_bot_drawdown_recovers():
    """Series that drops then recovers: trailing max-DD is the worst intra-window dip."""
    nav = _nav_series([100, 80, 100])
    # The worst point in the window was 80 from peak 100 → -20% max DD
    dd = per_bot_drawdown(nav, window_days=5)
    assert dd < -0.15  # at least 15% down (actual ~-20%)
    assert dd >= -0.25  # not worse than 25%


def test_per_bot_drawdown_deep():
    """Series with 30% drop in window → DD around -30%."""
    nav = _nav_series([100] + [70] * 5)
    dd = per_bot_drawdown(nav, window_days=10)
    assert dd < -0.25  # at least 25% down


# ---------------------------------------------------------------------------
# Should-pause tests
# ---------------------------------------------------------------------------

def test_should_pause_on_large_drawdown():
    """30-day drawdown > 25% → paused."""
    # 40% loss over the last 5 days (all within 30-day window)
    nav = _nav_series([100_000, 60_000, 60_000, 60_000])
    paused, reason = should_pause_bot("test-bot", nav)
    assert paused is True
    assert "drawdown" in reason.lower()


def test_should_not_pause_small_drawdown():
    """10% loss → not paused."""
    nav = _nav_series([100_000, 95_000, 90_000])
    paused, _ = should_pause_bot("test-bot", nav)
    assert paused is False


def test_should_not_pause_empty():
    """Empty NAV → not paused."""
    paused, reason = should_pause_bot("test-bot", [])
    assert paused is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Paper engine block_new_entries test
# ---------------------------------------------------------------------------

def test_paper_block_new_entries_no_open():
    """With block_new_entries=True, a brand-new position is NOT opened."""
    portfolio = Portfolio(bot_id="t", cash=10_000.0, positions={})
    weights = {"SPY": 0.5}
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}

    result = rebalance(portfolio, weights, prices, advs, as_of=date(2026, 5, 7), block_new_entries=True)

    assert "SPY" not in result.portfolio.positions
    assert any("blocked" in s for s in result.skipped)
    assert result.trades == []


def test_paper_block_new_entries_allows_existing():
    """With block_new_entries=True, an EXISTING position can still be trimmed."""
    portfolio = Portfolio(
        bot_id="t",
        cash=0.0,
        positions={"SPY": Position(symbol="SPY", shares=20, avg_cost=500.0)},
    )
    weights = {"SPY": 0.0}  # exit existing
    prices = {"SPY": 500.0}
    advs = {"SPY": 1e10}

    result = rebalance(portfolio, weights, prices, advs, as_of=date(2026, 5, 7), block_new_entries=True)

    # Existing SPY position should be sold
    assert any(t.side == "SELL" for t in result.trades)
