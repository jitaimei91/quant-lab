import json
from datetime import date

from quant_lab.reporting.dashboard import write_dashboard_data
from quant_lab.tournament.stats import Metrics


def test_write_dashboard_data_creates_leaderboard_json(tmp_path):
    out = tmp_path / "data"
    leaderboard = [
        ("spy-vol", Metrics(0.05, 0.10, 0.6, 0.15, -0.05, 100), {"SPY": 1.0}),
    ]
    nav_history = {"spy-vol": [(date(2026, 5, 5), 100_000), (date(2026, 5, 6), 101_000)]}
    market = {"SPY": {"change_pct": 0.31, "ytd_pct": 5.1},
              "QQQ": {"change_pct": 0.54, "ytd_pct": 8.7}}
    write_dashboard_data(out_dir=out, leaderboard=leaderboard, nav_history=nav_history,
                         market=market, generated_at=date(2026, 5, 6))

    leaderboard_json = json.loads((out / "leaderboard.json").read_text())
    assert leaderboard_json["bots"][0]["bot_id"] == "spy-vol"
    assert leaderboard_json["market"]["SPY"]["change_pct"] == 0.31

    nav_json = json.loads((out / "nav_history.json").read_text())
    assert nav_json["spy-vol"][-1]["nav"] == 101_000
