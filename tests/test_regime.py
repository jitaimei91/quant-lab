"""Tests for engine/regime.py — VIX kill-switch + per-bot drawdown halts."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from quant_lab.engine.hmm_regime import HMMState, save_hmm
from quant_lab.engine.regime import (
    _hmm_observations,
    hmm_regime_classify,
    per_bot_drawdown,
    regime_state,
    should_pause_bot,
)
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


# ---------------------------------------------------------------------------
# HMM regime tests
# ---------------------------------------------------------------------------

def _make_bar(symbol: str, close: float, dt: date | None = None) -> Bar:
    d = dt or date(2026, 1, 2)
    return Bar(symbol=symbol, date=d, open=close, high=close, low=close, close=close, volume=1_000_000)


def _make_series(symbol: str, closes: list[float]) -> list[Bar]:
    return [
        Bar(symbol=symbol, date=date(2025, 1, 1) + timedelta(days=i),
            open=c, high=c, low=c, close=c, volume=1_000_000)
        for i, c in enumerate(closes)
    ]


def _make_mock_hmm(n_states: int = 4) -> HMMState:
    """HMM with 4 states, means at VIX=10,20,30,50."""
    means = np.array([
        [10.0, 0.0, 0.05, 0.10, 0.0],
        [20.0, 0.0, 0.01, 0.15, 0.0],
        [30.0, 0.0, -0.02, 0.22, 0.0],
        [50.0, 0.0, -0.10, 0.40, 0.0],
    ])
    covs = np.array([np.eye(5) * 0.1 for _ in range(n_states)])
    transition = np.ones((n_states, n_states)) / n_states
    initial = np.ones(n_states) / n_states
    return HMMState(n_states=n_states, means=means, covariances=covs,
                    transition=transition, initial=initial)


def test_hmm_observations_shape():
    """_hmm_observations returns (lookback, 5) array."""
    vix_closes = [20.0 + i * 0.1 for i in range(100)]
    spy_closes = [400.0 + i * 0.5 for i in range(150)]
    histories = {
        "^VIX": _make_series("^VIX", vix_closes),
        "SPY": _make_series("SPY", spy_closes),
    }
    obs = _hmm_observations(histories, lookback=60)
    assert obs.shape == (60, 5)


def test_hmm_observations_correct_shape_minimal():
    """_hmm_observations always returns (lookback, 5) even with minimal data."""
    obs = _hmm_observations({}, lookback=30)
    assert obs.shape == (30, 5)
    assert np.all(obs == 0.0)


def test_hmm_observations_vix_populated():
    """VIX feature column is non-zero when VIX data is present."""
    vix_closes = [25.0] * 100
    histories = {"^VIX": _make_series("^VIX", vix_closes)}
    obs = _hmm_observations(histories, lookback=60)
    assert np.any(obs[:, 0] != 0.0)


def test_hmm_observations_missing_tlt_shy_zeros():
    """Feature 4 is zero when TLT/SHY are absent."""
    histories = {"^VIX": _make_series("^VIX", [20.0] * 100)}
    obs = _hmm_observations(histories, lookback=60)
    assert np.all(obs[:, 4] == 0.0)


def test_hmm_regime_classify_with_mocked_hmm(tmp_path):
    """hmm_regime_classify returns valid regime dict given a saved HMM."""
    hmm = _make_mock_hmm()
    hmm_path = tmp_path / "hmm.json"
    save_hmm(hmm, hmm_path)

    vix_closes = [20.0] * 100
    spy_closes = [400.0] * 150
    histories = {
        "^VIX": _make_series("^VIX", vix_closes),
        "SPY": _make_series("SPY", spy_closes),
    }
    result = hmm_regime_classify(histories, hmm_path)

    assert "regime_id" in result
    assert "regime_name" in result
    assert "regime_probs" in result
    assert "regime_confidence" in result
    assert result["regime_name"] in ("risk-on", "chop", "risk-off", "crisis")
    assert 0.0 <= result["regime_confidence"] <= 1.0


def test_hmm_regime_classify_missing_file_returns_default():
    """hmm_regime_classify with missing HMM path returns fallback dict."""
    result = hmm_regime_classify({}, Path("/tmp/does_not_exist_xyz.json"))
    assert result["regime_name"] in ("risk-on", "chop", "risk-off", "crisis")
    assert result["regime_confidence"] == 0.25  # uniform fallback


def test_regime_state_hmm_field_none_without_model():
    """regime_state includes hmm=None when no HMM file exists."""
    histories = {"^VIX": [_make_bar("^VIX", 25.0)]}
    state = regime_state(histories, hmm_state_path=Path("/tmp/no_such_hmm.json"))
    assert "hmm" in state
    assert state["hmm"] is None


def test_regime_state_hmm_field_populated_with_model(tmp_path):
    """regime_state includes hmm classification when model exists."""
    hmm = _make_mock_hmm()
    hmm_path = tmp_path / "hmm.json"
    save_hmm(hmm, hmm_path)

    vix_bars = _make_series("^VIX", [20.0] * 100)
    spy_bars = _make_series("SPY", [400.0] * 150)
    histories = {"^VIX": vix_bars, "SPY": spy_bars}

    state = regime_state(histories, hmm_state_path=hmm_path)
    assert state["hmm"] is not None
    assert "regime_name" in state["hmm"]


def test_regime_state_existing_tests_still_pass():
    """Core regime_state fields are unchanged by HMM additions."""
    state = regime_state({"^VIX": [_make_bar("^VIX", 25.0)]})
    assert state["regime"] == "NORMAL"
    assert state["halt_new_entries"] is False
    assert state["liquidate_all"] is False
    assert state["vix"] == 25.0
    assert "hmm" in state  # new field present but may be None
