import json

from quant_lab.persistence import (
    save_portfolios,
    load_portfolios,
    save_nav_history,
    load_nav_history,
    append_trades,
)
from quant_lab.types import Portfolio, Position, Trade
from datetime import date


def test_save_and_load_portfolios(tmp_path):
    p = Portfolio(
        bot_id="spy-vol",
        cash=5000.0,
        positions={"SPY": Position(symbol="SPY", shares=10, avg_cost=500.0)},
    )
    path = tmp_path / "portfolios.json"
    save_portfolios([p], path)
    loaded = load_portfolios(path)
    assert len(loaded) == 1
    assert loaded[0].bot_id == "spy-vol"
    assert loaded[0].positions["SPY"].shares == 10


def test_load_portfolios_missing_file_returns_empty(tmp_path):
    path = tmp_path / "absent.json"
    assert load_portfolios(path) == []


def test_save_and_load_nav_history(tmp_path):
    history = {"spy-vol": [(date(2026, 5, 5), 100_000.0), (date(2026, 5, 6), 101_000.0)]}
    path = tmp_path / "nav.json"
    save_nav_history(history, path)
    loaded = load_nav_history(path)
    assert loaded["spy-vol"][0] == (date(2026, 5, 5), 100_000.0)
    assert loaded["spy-vol"][-1] == (date(2026, 5, 6), 101_000.0)


def test_append_trades(tmp_path):
    path = tmp_path / "trades.jsonl"
    t = Trade(bot_id="spy-vol", symbol="SPY", side="BUY", shares=2.0, price=500.0,
              slippage_bps=5.0, timestamp=date(2026, 5, 6))
    append_trades([t], path)
    append_trades([t], path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["symbol"] == "SPY"
