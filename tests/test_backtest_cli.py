# tests/test_backtest_cli.py
import json
from datetime import date

from quant_lab.main import backtest_command
from quant_lab.types import Bar


def _synth(symbol, n=2200, drift=0.0004):
    base = date(2017, 1, 2)
    bars, price = [], 400.0
    for i in range(n):
        d = base.fromordinal(base.toordinal() + i)
        if d.weekday() >= 5:
            continue
        price *= (1 + drift)
        bars.append(Bar(symbol=symbol, date=d, open=price, high=price, low=price, close=price, volume=50_000_000))
    return bars


def test_backtest_command_writes_calibration_artifacts(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}
    monkeypatch.setattr("quant_lab.main.fetch_history", lambda symbol, lookback_days=365: histories.get(symbol.upper(), []))

    backtest_command(
        out_dir=tmp_path / "backtest",
        start=date(2017, 1, 1),
        end=date(2024, 1, 1),
        train_years=3,
        step_months=12,
        enable_slippage_sweep=False,
        enable_regime_stress=False,
    )

    payload = json.loads((tmp_path / "backtest" / "backtest_results.json").read_text())
    assert "strategies" in payload
    bot_ids = {s["bot_id"] for s in payload["strategies"]}
    assert "spy-vol" in bot_ids
    assert "qqq-vol" in bot_ids
