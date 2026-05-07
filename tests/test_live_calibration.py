"""Tests for live calibration weight updates."""
from __future__ import annotations

import json
import pytest
from datetime import date, timedelta

from quant_lab.ensemble.live_calibration import update_weights_from_live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_DATE = date(2025, 1, 2)


def _make_nav_series(n_days: int, drift: float = 0.0004, seed: int = 0) -> list[tuple[date, float]]:
    """Generate synthetic (date, nav) pairs."""
    import random
    rng = random.Random(seed)
    series = []
    nav = 1.0
    for i in range(n_days):
        d = _BASE_DATE + timedelta(days=i)
        nav *= 1 + drift + rng.gauss(0.0, 0.01)
        nav = max(nav, 0.01)
        series.append((d, nav))
    return series


def _make_spy_rets(n_days: int, seed: int = 42) -> list[float]:
    import random
    rng = random.Random(seed)
    return [rng.gauss(0.0003, 0.01) for _ in range(n_days)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_insufficient_history_uses_fallback(tmp_path):
    """Bots with < min_days of data fall back to backtest-calibrated weights."""
    # Only 30 days of live NAV (< min_days=60)
    nav_history = {
        "spy-vol": _make_nav_series(30),
    }
    bench_rets = {"SPY": _make_spy_rets(30)}
    weights_path = tmp_path / "live_weights.json"

    # Provide a minimal backtest fallback
    bt_path = tmp_path / "backtest_results.json"
    bt_data = {
        "strategies": [
            {
                "bot_id": "spy-vol",
                "aggregate": {
                    "sharpe": 0.8,
                    "sharpe_ci_lo": 0.1,
                    "significance_weight": 0.3,
                },
                "per_window": [{"sharpe": 0.8}],
            }
        ]
    }
    bt_path.write_text(json.dumps(bt_data), encoding="utf-8")

    result = update_weights_from_live(
        nav_history=nav_history,
        benchmark_returns=bench_rets,
        min_days=60,
        weights_path=weights_path,
        backtest_weights_path=bt_path,
        n_iter=200,
    )

    # spy-vol has fallback data → should appear in result
    assert "spy-vol" in result
    assert result["spy-vol"] > 0


def test_sufficient_history_uses_live_weights(tmp_path):
    """Bots with >= min_days of live data get live-Sharpe-computed weights."""
    nav_history = {
        "bot-a": _make_nav_series(80, drift=0.0006, seed=1),
        "bot-b": _make_nav_series(80, drift=0.0001, seed=2),
    }
    spy_rets = _make_spy_rets(80)
    bench_rets = {"SPY": spy_rets}
    weights_path = tmp_path / "live_weights.json"

    result = update_weights_from_live(
        nav_history=nav_history,
        benchmark_returns=bench_rets,
        min_days=60,
        weights_path=weights_path,
        n_iter=200,
    )

    assert isinstance(result, dict)
    # Should produce some weights
    assert len(result) > 0
    # Weights should sum to ~1.0
    total = sum(result.values())
    assert total == pytest.approx(1.0, abs=0.01) or total == pytest.approx(0.0, abs=0.01)


def test_live_weights_file_written(tmp_path):
    """update_weights_from_live writes live_weights.json at the specified path."""
    nav_history = {
        "spy-vol": _make_nav_series(70, seed=10),
    }
    bench_rets = {"SPY": _make_spy_rets(70)}
    weights_path = tmp_path / "backtest" / "live_weights.json"

    update_weights_from_live(
        nav_history=nav_history,
        benchmark_returns=bench_rets,
        min_days=60,
        weights_path=weights_path,
        n_iter=200,
    )

    assert weights_path.exists(), "live_weights.json was not written"
    data = json.loads(weights_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_meta_ensemble_excluded_from_live_calibration(tmp_path):
    """meta-ensemble itself should never appear in live_weights.json."""
    nav_history = {
        "spy-vol": _make_nav_series(70, seed=20),
        "meta-ensemble": _make_nav_series(70, seed=21),  # should be excluded
    }
    bench_rets = {"SPY": _make_spy_rets(70)}
    weights_path = tmp_path / "live_weights.json"

    result = update_weights_from_live(
        nav_history=nav_history,
        benchmark_returns=bench_rets,
        min_days=60,
        weights_path=weights_path,
        n_iter=200,
    )

    assert "meta-ensemble" not in result
