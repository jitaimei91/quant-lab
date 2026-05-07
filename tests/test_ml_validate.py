"""Tests for ML validation gates."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.ml.validate import (
    label_shuffle_test,
    oos_stability_test,
    run_all_gates,
    walk_forward_sharpe_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars(symbol, n=520, trend=0.0003, vol=0.012, seed=42):
    from quant_lab.types import Bar
    import random

    rng = random.Random(seed)
    start = date(2020, 1, 6)
    price = 100.0
    bars = []
    for i in range(n):
        ret = rng.gauss(trend, vol)
        price = max(price * (1 + ret), 0.01)
        bars.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=i),
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=2_000_000,
            )
        )
    return bars


def _make_X_y_clean(n: int = 200, n_features: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where y has NO relationship to X — pure noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.standard_normal(n), name="fwd_return")
    return X, y


def _make_X_y_leaked(n: int = 200, n_features: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X contains the future label directly (blatant lookahead)."""
    rng = np.random.default_rng(seed)
    X_noise = rng.standard_normal((n, n_features - 1))
    y_vals = rng.standard_normal(n)
    # Feature 0 = the label itself (blatant lookahead leakage)
    X = pd.DataFrame(
        np.column_stack([y_vals.reshape(-1, 1), X_noise]),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(y_vals, name="fwd_return")
    return X, y


def _simple_train_fn(X: pd.DataFrame, y: pd.Series) -> Any:
    """A simple linear regressor as train_fn for tests."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0)
    model.fit(X.fillna(0.0).values, y.values)
    return model


def _leaked_train_fn(X: pd.DataFrame, y: pd.Series) -> Any:
    """Ridge trained on X that includes the label as feature 0."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=0.0001)  # tiny regularisation so it picks up the leakage
    model.fit(X.fillna(0.0).values, y.values)
    return model


# ---------------------------------------------------------------------------
# label_shuffle_test
# ---------------------------------------------------------------------------


def test_label_shuffle_clean_data_passes():
    """With no relationship between X and y, shuffle test should PASS (near-zero Sharpe)."""
    X, y = _make_X_y_clean(n=300, seed=7)
    result = label_shuffle_test(
        train_fn=_simple_train_fn,
        X=X,
        y=y,
        n_shuffles=5,
        seed=1,
    )
    assert isinstance(result["pass"], bool)
    # Clean data: mean shuffled Sharpe should be near zero → pass
    assert result["pass"], f"Expected PASS on clean data, got {result}"


def test_label_shuffle_leaked_data_fails():
    """With blatant lookahead leakage (feature = label), shuffle test should FAIL.

    When feature 0 = true label value, a model trained on SHUFFLED labels will
    learn to use feature 0 (which equals the *true* label, not the shuffled one).
    Evaluating on held-out true returns thus yields non-trivial Sharpe, triggering FAIL.
    """
    X, y = _make_X_y_leaked(n=600, seed=3)

    # Verify the leakage: in-sample R² should be high
    model_clean = _leaked_train_fn(X, y)
    from sklearn.metrics import r2_score
    r2 = r2_score(y.values, model_clean.predict(X.values))
    assert r2 > 0.5, f"Expected strong in-sample fit with leaked feature, r2={r2}"

    # Shuffle test should detect the leakage
    result = label_shuffle_test(
        train_fn=_leaked_train_fn,
        X=X,
        y=y,
        n_shuffles=10,
        seed=42,
    )
    # With leakage, shuffled-label model still predicts OOS true returns → FAIL
    assert not result["pass"], (
        f"Expected FAIL on leaked data, mean_shuffled={result['mean_shuffled']:.4f}. "
        f"detail: {result['detail']}"
    )


def test_label_shuffle_result_structure():
    X, y = _make_X_y_clean(n=100, seed=0)
    result = label_shuffle_test(
        train_fn=_simple_train_fn, X=X, y=y, n_shuffles=3, seed=0
    )
    assert "shuffled_sharpes" in result
    assert "mean_shuffled" in result
    assert "std_shuffled" in result
    assert "pass" in result
    assert "detail" in result
    assert len(result["shuffled_sharpes"]) == 3


# ---------------------------------------------------------------------------
# oos_stability_test
# ---------------------------------------------------------------------------


def test_oos_stability_passes_when_live_in_range():
    result = oos_stability_test(
        backtest_fold_sharpes=[1.0, 1.2, 0.9, 1.1],
        live_window_sharpe=1.05,
        tolerance=0.30,
    )
    assert result["pass"] is True


def test_oos_stability_fails_when_live_far_outside():
    result = oos_stability_test(
        backtest_fold_sharpes=[1.0, 1.2, 0.9, 1.1],
        live_window_sharpe=-2.0,  # way outside
        tolerance=0.30,
    )
    assert result["pass"] is False


def test_oos_stability_passes_when_no_live_data():
    result = oos_stability_test(
        backtest_fold_sharpes=[1.0, 1.2],
        live_window_sharpe=float("nan"),
    )
    assert result["pass"] is True


def test_oos_stability_passes_when_no_backtest():
    result = oos_stability_test(
        backtest_fold_sharpes=[],
        live_window_sharpe=0.5,
    )
    assert result["pass"] is True


# ---------------------------------------------------------------------------
# walk_forward_sharpe_gate
# ---------------------------------------------------------------------------


def test_wf_sharpe_gate_passes_when_bot_beats_benchmark():
    result = walk_forward_sharpe_gate(
        bot_fold_sharpes=[0.8, 1.0, 0.9],
        benchmark_fold_sharpes=[0.3, 0.4, 0.35],
    )
    assert result["pass"] is True


def test_wf_sharpe_gate_fails_when_bot_below_benchmark():
    result = walk_forward_sharpe_gate(
        bot_fold_sharpes=[0.2, 0.1, 0.15],
        benchmark_fold_sharpes=[0.5, 0.6, 0.55],
    )
    assert result["pass"] is False


def test_wf_sharpe_gate_fails_when_no_folds():
    result = walk_forward_sharpe_gate(
        bot_fold_sharpes=[],
        benchmark_fold_sharpes=[0.5],
    )
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# run_all_gates
# ---------------------------------------------------------------------------


def test_run_all_gates_passes_when_all_pass():
    artifacts = {
        "label_shuffle_result": {"pass": True, "detail": "ok"},
        "fold_sharpes": [1.0, 1.2, 0.9],
        "benchmark_fold_sharpes": [0.3, 0.4],
        "live_sharpe": None,  # no live data → stability defaults to pass
    }
    result = run_all_gates("test-bot", artifacts)
    assert result["overall_pass"] is True
    assert result["reasons_failed"] == []
    assert result["bot_id"] == "test-bot"


def test_run_all_gates_fails_when_any_gate_fails():
    artifacts = {
        "label_shuffle_result": {"pass": False, "detail": "FAIL — leakage"},
        "fold_sharpes": [1.0, 1.2],
        "benchmark_fold_sharpes": [0.3],
        "live_sharpe": None,
    }
    result = run_all_gates("test-bot", artifacts)
    assert result["overall_pass"] is False
    assert "label_shuffle" in result["reasons_failed"]


def test_run_all_gates_result_structure():
    artifacts = {
        "label_shuffle_result": {"pass": True, "detail": "ok"},
        "fold_sharpes": [0.8],
        "benchmark_fold_sharpes": [0.4],
        "live_sharpe": None,
    }
    result = run_all_gates("gradboost", artifacts)
    assert "bot_id" in result
    assert "gates" in result
    assert "overall_pass" in result
    assert "reasons_failed" in result
    assert set(result["gates"].keys()) == {"label_shuffle", "oos_stability", "walk_forward_sharpe"}
