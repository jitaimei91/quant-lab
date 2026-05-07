# tests/test_backtest_e2e.py
"""E2E smoke: pull synthetic 7 years of data, run walk-forward, write report, assert artifacts exist."""
import json
from datetime import date
from pathlib import Path

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


def test_backtest_e2e_produces_calibration_artifacts(tmp_path, monkeypatch):
    histories = {"SPY": _synth("SPY"), "QQQ": _synth("QQQ", drift=0.0006)}
    monkeypatch.setattr("quant_lab.main.fetch_history", lambda symbol, lookback_days=365: histories.get(symbol.upper(), []))

    out = tmp_path / "bt"
    backtest_command(
        out_dir=out,
        start=date(2017, 1, 1),
        end=date(2023, 1, 1),
        train_years=3,
        step_months=12,
        enable_slippage_sweep=True,
        enable_regime_stress=False,
    )

    assert (out / "backtest_results.json").exists()
    assert (out / "backtest_curves.json").exists()
    assert (out / "calibration_report.md").exists()

    payload = json.loads((out / "backtest_results.json").read_text())
    bot_ids = {s["bot_id"] for s in payload["strategies"]}
    assert {"spy-vol", "qqq-vol"} <= bot_ids
    # Slippage sweep should be present
    assert payload["slippage_sweep"] is not None
    assert "1.0" in payload["slippage_sweep"]
