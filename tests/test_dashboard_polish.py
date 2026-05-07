"""Tests for Phase 4 dashboard polish exporters."""
import json
from datetime import date

import pytest

from quant_lab.reporting.dashboard import (
    write_dashboard_data,
    write_per_bot_files,
    write_validation_data,
)
from quant_lab.tournament.stats import Metrics


_METRICS = Metrics(
    total_return=0.12,
    annualized_return=0.10,
    sharpe=0.85,
    sharpe_ci_lo=0.20,
    sharpe_ci_hi=1.50,
    volatility=0.18,
    max_drawdown=-0.08,
    days=252,
    alpha_t_stat_vs_spy=1.2,
    alpha_t_stat_vs_qqq=0.9,
    significance_weight=0.75,
    factor_loadings={"MKT": 0.8, "SIZE": 0.1, "VALUE": -0.2},
)

_NAV = {
    "bot-a": [
        (date(2026, 1, 2), 100_000.0),
        (date(2026, 1, 3), 101_000.0),
        (date(2026, 1, 6), 102_000.0),
    ]
}


def test_write_per_bot_files_creates_bot_json(tmp_path):
    leaderboard = [("bot-a", _METRICS, {"SPY": 0.6, "QQQ": 0.4})]
    write_per_bot_files(tmp_path, leaderboard, _NAV)
    bot_file = tmp_path / "bots" / "bot-a.json"
    assert bot_file.exists(), "bot-a.json should be written"
    data = json.loads(bot_file.read_text())
    assert data["bot_id"] == "bot-a"
    assert data["metrics"]["sharpe"] == pytest.approx(0.85)
    assert len(data["nav_series"]) == 3
    assert len(data["daily_returns"]) == 2  # one per adjacent pair


def test_write_per_bot_files_embedded_in_write_dashboard_data(tmp_path):
    leaderboard = [("bot-a", _METRICS, {"SPY": 1.0})]
    market = {"SPY": {"change_pct": 0.1, "ytd_pct": 2.0}, "QQQ": {"change_pct": 0.2, "ytd_pct": 3.0}}
    write_dashboard_data(
        out_dir=tmp_path,
        leaderboard=leaderboard,
        nav_history=_NAV,
        market=market,
        generated_at=date(2026, 5, 7),
    )
    bot_file = tmp_path / "bots" / "bot-a.json"
    assert bot_file.exists(), "write_dashboard_data should call write_per_bot_files"


def test_write_validation_data_creates_validation_json(tmp_path):
    backtest_data = {
        "strategies": [
            {
                "bot_id": "momo",
                "aggregate": {
                    "sharpe": 1.1,
                    "sharpe_ci_lo": 0.3,
                    "sharpe_ci_hi": 1.9,
                    "median_alpha_t": 0.8,
                    "significance_weight": 0.75,
                    "windows_evaluated": 4,
                    "total_test_days": 1000,
                },
                "per_window": [],
            },
            {
                "bot_id": "breakout",
                "aggregate": {
                    "sharpe": -0.1,
                    "sharpe_ci_lo": -0.8,
                    "sharpe_ci_hi": 0.6,
                    "median_alpha_t": -0.2,
                    "significance_weight": 0.0,
                    "windows_evaluated": 4,
                    "total_test_days": 1000,
                },
                "per_window": [],
            },
        ]
    }
    bt_path = tmp_path / "backtest_results.json"
    bt_path.write_text(json.dumps(backtest_data))

    write_validation_data(out_dir=tmp_path, backtest_results_path=bt_path)

    val_file = tmp_path / "validation.json"
    assert val_file.exists()
    data = json.loads(val_file.read_text())
    by_bot = {s["bot_id"]: s for s in data["strategies"]}
    assert by_bot["momo"]["significance_badge"] == "green"
    assert by_bot["breakout"]["significance_badge"] == "gray"
    assert by_bot["breakout"]["failed_validation"] is True
    assert by_bot["momo"]["failed_validation"] is False


def test_write_validation_data_no_op_when_file_missing(tmp_path):
    missing = tmp_path / "no_such_file.json"
    # Should not raise
    write_validation_data(out_dir=tmp_path, backtest_results_path=missing)
    assert not (tmp_path / "validation.json").exists()


def test_write_validation_data_merges_lifecycle_state(tmp_path):
    """When lifecycle_state is provided, each strategy entry has a lifecycle key."""
    from quant_lab.lifecycle import LifecycleState
    from datetime import date as _date

    backtest_data = {
        "strategies": [
            {
                "bot_id": "momo",
                "aggregate": {"sharpe": 1.1, "significance_weight": 0.75},
                "per_window": [],
            },
            {
                "bot_id": "breakout",
                "aggregate": {"sharpe": -0.1, "significance_weight": 0.0},
                "per_window": [],
            },
        ]
    }
    bt_path = tmp_path / "backtest_results.json"
    bt_path.write_text(json.dumps(backtest_data))

    lifecycle_state = {
        "momo": LifecycleState(bot_id="momo", paused=False),
        "breakout": LifecycleState(
            bot_id="breakout",
            paused=True,
            paused_at=_date(2025, 6, 1),
            pause_reason="low significance for 90d",
            consecutive_fail_days=95,
        ),
    }

    write_validation_data(
        out_dir=tmp_path,
        backtest_results_path=bt_path,
        lifecycle_state=lifecycle_state,
    )

    data = json.loads((tmp_path / "validation.json").read_text())
    by_bot = {s["bot_id"]: s for s in data["strategies"]}

    assert by_bot["momo"]["lifecycle"]["paused"] is False
    assert by_bot["breakout"]["lifecycle"]["paused"] is True
    assert by_bot["breakout"]["lifecycle"]["consecutive_fail_days"] == 95
    assert "low significance" in by_bot["breakout"]["lifecycle"]["pause_reason"]


def test_write_validation_data_lifecycle_none_omits_key(tmp_path):
    """When lifecycle_state is None, the lifecycle key is present but null."""
    backtest_data = {
        "strategies": [
            {
                "bot_id": "momo",
                "aggregate": {"sharpe": 1.1, "significance_weight": 0.75},
                "per_window": [],
            },
        ]
    }
    bt_path = tmp_path / "backtest_results.json"
    bt_path.write_text(json.dumps(backtest_data))

    write_validation_data(out_dir=tmp_path, backtest_results_path=bt_path)

    data = json.loads((tmp_path / "validation.json").read_text())
    entry = data["strategies"][0]
    # lifecycle key exists but is null when not provided
    assert "lifecycle" in entry
    assert entry["lifecycle"] is None
