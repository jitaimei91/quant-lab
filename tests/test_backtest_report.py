# tests/test_backtest_report.py
import json
from datetime import date
from pathlib import Path

from quant_lab.backtest.report import write_calibration_report
from quant_lab.backtest.harness import WalkForwardResult


def test_write_calibration_report_emits_json_and_markdown(tmp_path: Path):
    result = WalkForwardResult(
        nav_by_window={"wf-2020": {"strat-a": [100_000, 102_000, 105_000]}},
        dates_by_window={"wf-2020": [date(2020, 1, 1), date(2020, 6, 1), date(2020, 12, 31)]},
        returns_by_window={"wf-2020": {"strat-a": [0.02, 0.029]}},
    )
    benchmark_returns = {"wf-2020": [0.005, 0.010]}
    out_dir = tmp_path / "out"
    write_calibration_report(
        out_dir=out_dir,
        wf_result=result,
        benchmark_returns_by_window=benchmark_returns,
        slippage_sweep=None,
        regime_results={},
    )
    payload = json.loads((out_dir / "backtest_results.json").read_text())
    assert "strategies" in payload
    assert any(s["bot_id"] == "strat-a" for s in payload["strategies"])
    md = (out_dir / "calibration_report.md").read_text()
    assert "# Calibration Report" in md
    assert "strat-a" in md
