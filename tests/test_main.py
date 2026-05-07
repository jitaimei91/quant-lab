from datetime import date
from unittest.mock import patch

from quant_lab.main import morning_command


def test_morning_command_dry_run(tmp_path, monkeypatch):
    """End-to-end smoke test: morning_command should not crash on synthetic data."""
    from quant_lab.types import Bar
    base = date(2026, 1, 2)

    def fake_fetch(symbol, lookback_days=365):
        bars = []
        price = 500.0
        for i in range(120):
            d = base.fromordinal(base.toordinal() + i)
            price *= 1.0005
            bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=10_000_000))
        return bars

    monkeypatch.setattr("quant_lab.main.fetch_history", fake_fetch)

    state_dir = tmp_path / "state"
    dashboard_dir = tmp_path / "dashboard_data"
    snapshot_dir = tmp_path / "snapshots"

    with patch("quant_lab.main.post_to_discord"):
        morning_command(
            state_dir=state_dir,
            dashboard_data_dir=dashboard_dir,
            snapshot_dir=snapshot_dir,
            discord_webhook=None,  # skip Discord
            dashboard_url=None,
        )

    assert (state_dir / "portfolios.json").exists()
    assert (state_dir / "nav_history.json").exists()
    assert (dashboard_dir / "leaderboard.json").exists()
