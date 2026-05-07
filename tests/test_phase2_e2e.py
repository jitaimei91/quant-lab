"""Phase 2 end-to-end integration test.

Runs morning_command with synthetic multi-symbol data (including factor proxies)
and verifies that:
  1. >= 8 bots have NAV recorded
  2. leaderboard.json includes the new Phase 2 Metrics fields
  3. state/last_morning.json is written
  4. No crash across two consecutive morning runs
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import patch

from quant_lab.main import morning_command
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2025, 1, 2)
_N = 400  # enough history for momo (126d), ma_cross (200d), etc.


def _synth(
    symbol: str,
    n: int = _N,
    drift: float = 0.0004,
    volatility: float = 0.01,
    base_price: float = 100.0,
    volume: int = 50_000_000,
    seed: int = 0,
) -> list[Bar]:
    """Generate n bars with geometric drift + random walk."""
    import random
    rng = random.Random(seed + hash(symbol) % 1000)
    bars = []
    price = base_price
    for i in range(n):
        d = _BASE_DATE + timedelta(days=i)
        ret = drift + rng.gauss(0.0, volatility)
        price = max(price * (1 + ret), 0.01)
        bars.append(
            Bar(
                symbol=symbol,
                date=d,
                open=price * 0.999,
                high=price * 1.002,
                low=price * 0.997,
                close=price,
                volume=volume,
            )
        )
    return bars


def _make_histories() -> dict[str, list[Bar]]:
    """Synthetic histories covering all factor proxies + some stocks."""
    return {
        "SPY": _synth("SPY", drift=0.0004, base_price=500.0, seed=1),
        "QQQ": _synth("QQQ", drift=0.0005, base_price=430.0, seed=2),
        "IWM": _synth("IWM", drift=0.0003, base_price=200.0, seed=3),
        "VTV": _synth("VTV", drift=0.0003, base_price=160.0, seed=4),
        "VUG": _synth("VUG", drift=0.0004, base_price=310.0, seed=5),
        "AAPL": _synth("AAPL", drift=0.0006, base_price=180.0, seed=6),
        "NVDA": _synth("NVDA", drift=0.0008, base_price=700.0, seed=7),
        # ^VIX with low levels (NORMAL regime)
        "^VIX": _synth("^VIX", drift=-0.0001, base_price=18.0, volatility=0.005, volume=0, seed=8),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_phase2_morning_command_all_bots(tmp_path):
    """All 9 registered strategies run and produce NAV entries."""
    histories = _make_histories()

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol, [])

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    with patch("quant_lab.main.fetch_history", side_effect=fake_fetch), \
         patch("quant_lab.main.post_to_discord"):
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url=None)

    nav_json = json.loads((dash / "nav_history.json").read_text())
    # At least 8 bots should have NAV entries (codex-native may be empty if ETF
    # universe has insufficient history)
    bots_with_nav = [bot_id for bot_id, series in nav_json.items() if series]
    assert len(bots_with_nav) >= 8, f"Expected >= 8 bots with NAV, got {len(bots_with_nav)}: {bots_with_nav}"


def test_phase2_leaderboard_has_new_fields(tmp_path):
    """leaderboard.json includes Phase 2 Metrics fields."""
    histories = _make_histories()

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol, [])

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    with patch("quant_lab.main.fetch_history", side_effect=fake_fetch), \
         patch("quant_lab.main.post_to_discord"):
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url=None)

    leaderboard = json.loads((dash / "leaderboard.json").read_text())
    bots = leaderboard["bots"]
    assert len(bots) > 0

    for bot in bots:
        m = bot["metrics"]
        assert "sharpe_ci_lo" in m, f"sharpe_ci_lo missing in {bot['bot_id']}"
        assert "sharpe_ci_hi" in m, f"sharpe_ci_hi missing in {bot['bot_id']}"
        assert "alpha_t_stat_vs_spy" in m, f"alpha_t_stat_vs_spy missing in {bot['bot_id']}"
        assert "alpha_t_stat_vs_qqq" in m, f"alpha_t_stat_vs_qqq missing in {bot['bot_id']}"
        assert "significance_weight" in m, f"significance_weight missing in {bot['bot_id']}"
        assert "factor_loadings" in m, f"factor_loadings missing in {bot['bot_id']}"


def test_phase2_last_morning_json_written(tmp_path):
    """state/last_morning.json is written with status=success."""
    histories = _make_histories()

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol, [])

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    with patch("quant_lab.main.fetch_history", side_effect=fake_fetch), \
         patch("quant_lab.main.post_to_discord"):
        morning_command(state, dash, snap, discord_webhook=None, dashboard_url=None)

    last_morning = json.loads((state / "last_morning.json").read_text())
    assert last_morning["status"] == "success"
    assert "timestamp" in last_morning
    assert isinstance(last_morning["strategies"], list)
    assert len(last_morning["strategies"]) > 0


def test_phase2_two_consecutive_runs(tmp_path):
    """Two consecutive morning runs succeed without crash or state corruption."""
    histories = _make_histories()

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol, [])

    state = tmp_path / "state"
    dash = tmp_path / "dash"
    snap = tmp_path / "snap"

    with patch("quant_lab.main.fetch_history", side_effect=fake_fetch), \
         patch("quant_lab.main.post_to_discord"):
        morning_command(state, dash, snap, discord_webhook=None, dashboard_url=None)
        morning_command(state, dash, snap, discord_webhook=None, dashboard_url=None)

    nav_json = json.loads((dash / "nav_history.json").read_text())
    # After 2 runs with same-day date, each bot should still have exactly 1 NAV entry
    for bot_id, series in nav_json.items():
        if series:
            assert len(series) == 1, f"{bot_id} should have 1 NAV entry (same day dedup)"
