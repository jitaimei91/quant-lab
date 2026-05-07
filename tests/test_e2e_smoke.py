"""End-to-end smoke test using synthetic data.

Runs the entire morning pipeline against fake yfinance data and verifies
state files, dashboard data, and a (mocked) Discord post all happen.
"""
import json
from datetime import date
from unittest.mock import patch

from quant_lab.main import morning_command
from quant_lab.types import Bar


def _synth(symbol, n=200, drift=0.0004):
    base = date(2026, 1, 2)
    bars, price = [], 500.0 if symbol == "SPY" else 450.0
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=50_000_000))
    return bars


def test_two_consecutive_morning_runs(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}

    def fake_fetch(symbol, lookback_days=365):
        return histories.get(symbol.upper(), [])

    monkeypatch.setattr("quant_lab.main.fetch_history", fake_fetch)

    state = tmp_path / "state"
    dash = tmp_path / "dashboard_data"
    snap = tmp_path / "snapshots"

    with patch("quant_lab.main.post_to_discord") as mp:
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url="https://y")
        assert mp.called

        # Second run on the same data must not crash
        morning_command(state, dash, snap, discord_webhook="https://x", dashboard_url="https://y")

    leaderboard = json.loads((dash / "leaderboard.json").read_text())
    bot_ids = {row["bot_id"] for row in leaderboard["bots"]}
    assert "spy-vol" in bot_ids
    assert "qqq-vol" in bot_ids
    assert leaderboard["market"]["SPY"]["change_pct"] != 0  # synthetic drift > 0
