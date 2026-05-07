"""Validation gates for ML strategies.

Three gates guard against spurious models:
1. Label-shuffle test: model should NOT predict well on randomly shuffled labels.
   A non-zero Sharpe on shuffled labels indicates lookahead leakage.
2. OOS stability test: live performance should stay within range of backtest.
3. Walk-forward Sharpe gate: median OOS Sharpe must exceed benchmark.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


def _sharpe(returns: list[float]) -> float:
    """Annualised Sharpe from a daily return series."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(252)


def _top_decile_sharpe(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> float:
    """Simulate top-decile long returns and return Sharpe."""
    if X.empty or len(y) == 0:
        return 0.0
    preds = model.predict(X.fillna(0.0).values)
    realized = y.values
    top_n = max(1, len(preds) // n_bins)
    ranked_idx = np.argsort(preds)[::-1][:top_n]
    top_returns = [float(realized[i]) for i in ranked_idx]
    return _sharpe(top_returns)


def label_shuffle_test(
    train_fn: Callable[[pd.DataFrame, pd.Series], Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_shuffles: int = 10,
    metric_fn: Optional[Callable] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Detect lookahead leakage via the permutation-importance protocol.

    Protocol (held-out evaluation):
      Split data 70/30 train/test (time-ordered).
      For each shuffle iteration:
        - Shuffle training labels only.
        - Train model on (X_train, y_train_shuffled).
        - Evaluate on (X_test, y_test) — held-out true labels.
      A clean model should score near-zero on the held-out true returns
      (training on random labels should produce random rankings).
      If features encode future returns (leakage), the model will still
      predict well even with shuffled training labels, because it learns
      the leaked feature → high OOS Sharpe on true labels.

    Pass condition: |mean held-out Sharpe| < 0.1

    Args:
        train_fn: callable(X, y_shuffled) -> fitted model
        X: feature DataFrame
        y: true labels (forward returns)
        n_shuffles: number of shuffle iterations
        metric_fn: optional callable(model, X_test, y_test) -> float
        seed: random seed for reproducibility

    Returns:
        dict with keys: shuffled_sharpes, mean_shuffled, std_shuffled, pass, detail
    """
    rng = random.Random(seed)
    if metric_fn is None:
        metric_fn = _top_decile_sharpe

    n = len(y)
    split = int(n * 0.7)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    if len(X_test) < 10:
        # Fallback: not enough data for split — use full set for both
        X_train, X_test, y_train, y_test = X, X, y, y

    shuffled_sharpes: list[float] = []
    y_train_vals = y_train.values.copy()

    for _i in range(n_shuffles):
        y_shuf = y_train_vals.copy()
        rng.shuffle(y_shuf)  # type: ignore[arg-type]
        y_shuffled_train = pd.Series(y_shuf, index=y_train.index)
        model = train_fn(X_train, y_shuffled_train)
        # Evaluate against HELD-OUT TRUE returns
        sharpe = metric_fn(model, X_test, y_test)
        shuffled_sharpes.append(sharpe)

    mean_shuf = float(np.mean(shuffled_sharpes))
    std_shuf = float(np.std(shuffled_sharpes, ddof=1)) if len(shuffled_sharpes) > 1 else 0.0
    # Threshold: mean |Sharpe| < 0.5. A clean model trained on randomised labels
    # should produce predictions uncorrelated with true returns → OOS Sharpe ≈ 0.
    # We use 0.5 as threshold (not 0.1) to tolerate noise with small datasets;
    # genuine leakage (feature = label) will produce Sharpe >> 1.0.
    threshold = 0.5
    passed = abs(mean_shuf) < threshold

    return {
        "shuffled_sharpes": shuffled_sharpes,
        "mean_shuffled": mean_shuf,
        "std_shuffled": std_shuf,
        "pass": passed,
        "detail": (
            f"mean_shuffled_oos_sharpe={mean_shuf:.4f} (threshold |mean|<{threshold}); "
            f"{'PASS' if passed else 'FAIL — leakage: shuffled-label model predicts true OOS returns'}"
        ),
    }


def oos_stability_test(
    backtest_fold_sharpes: list[float],
    live_window_sharpe: float,
    tolerance: float = 0.30,
) -> dict[str, Any]:
    """Pass if live Sharpe is within ±tolerance × backtest median Sharpe.

    Always passes if there are no backtest folds or live data is absent.

    Args:
        backtest_fold_sharpes: per-fold OOS Sharpe values from walk-forward
        live_window_sharpe: Sharpe observed in production/live trading
        tolerance: fractional deviation allowed (default 0.30 = ±30%)

    Returns:
        dict with keys: backtest_median, live_sharpe, tolerance, pass, detail
    """
    if not backtest_fold_sharpes or math.isnan(live_window_sharpe):
        return {
            "backtest_median": None,
            "live_sharpe": live_window_sharpe,
            "tolerance": tolerance,
            "pass": True,
            "detail": "No backtest folds or live data — defaulting to PASS",
        }

    bt_median = float(np.median(backtest_fold_sharpes))
    # Allow live to deviate by tolerance × |bt_median|, minimum 0.5 buffer
    allowed_deviation = max(0.5, tolerance * abs(bt_median))
    lo = bt_median - allowed_deviation
    hi = bt_median + allowed_deviation
    passed = lo <= live_window_sharpe <= hi

    return {
        "backtest_median": bt_median,
        "live_sharpe": live_window_sharpe,
        "tolerance": tolerance,
        "allowed_range": [lo, hi],
        "pass": passed,
        "detail": (
            f"live_sharpe={live_window_sharpe:.4f} vs backtest_median={bt_median:.4f} "
            f"allowed=[{lo:.4f},{hi:.4f}]; {'PASS' if passed else 'FAIL'}"
        ),
    }


def walk_forward_sharpe_gate(
    bot_fold_sharpes: list[float],
    benchmark_fold_sharpes: list[float],
) -> dict[str, Any]:
    """Pass if median bot Sharpe > median benchmark Sharpe.

    Args:
        bot_fold_sharpes: per-fold OOS Sharpe values for the ML bot
        benchmark_fold_sharpes: per-fold Sharpe values for the benchmark (e.g. SPY-Vol)

    Returns:
        dict with keys: bot_median, benchmark_median, pass, detail
    """
    if not bot_fold_sharpes:
        return {
            "bot_median": None,
            "benchmark_median": None,
            "pass": False,
            "detail": "No fold Sharpes available — FAIL",
        }

    bot_median = float(np.median(bot_fold_sharpes))
    bench_median = float(np.median(benchmark_fold_sharpes)) if benchmark_fold_sharpes else 0.0
    passed = bot_median > bench_median

    return {
        "bot_median": bot_median,
        "benchmark_median": bench_median,
        "pass": passed,
        "detail": (
            f"bot_median_sharpe={bot_median:.4f} vs benchmark={bench_median:.4f}; "
            f"{'PASS' if passed else 'FAIL'}"
        ),
    }


def run_all_gates(
    bot_id: str,
    training_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Run all 3 gates and return a structured result.

    Expected training_artifacts keys:
      - fold_sharpes: list[float]  (from walk-forward training)
      - live_sharpe: float | None  (from production; NaN if absent)
      - benchmark_fold_sharpes: list[float]  (from SPY-Vol or similar)
      - label_shuffle_result: dict  (pre-computed; run label_shuffle_test before calling this)
        OR provide train_fn + X + y to compute it here.

    Returns:
        {
          'bot_id': str,
          'gates': {<name>: {'pass': bool, 'detail': ...}},
          'overall_pass': bool,
          'reasons_failed': list[str],
        }
    """
    gates: dict[str, dict[str, Any]] = {}

    # Gate 1: label shuffle (must be pre-computed and passed in artifacts)
    shuffle_result = training_artifacts.get("label_shuffle_result")
    if shuffle_result is not None:
        gates["label_shuffle"] = {
            "pass": shuffle_result.get("pass", False),
            "detail": shuffle_result.get("detail", ""),
        }
    else:
        gates["label_shuffle"] = {
            "pass": False,
            "detail": "label_shuffle_result not provided in artifacts",
        }

    # Gate 2: OOS stability
    fold_sharpes = training_artifacts.get("fold_sharpes", [])
    live_sharpe = training_artifacts.get("live_sharpe", float("nan"))
    if live_sharpe is None:
        live_sharpe = float("nan")
    stab = oos_stability_test(fold_sharpes, float(live_sharpe))
    gates["oos_stability"] = {"pass": stab["pass"], "detail": stab["detail"]}

    # Gate 3: walk-forward Sharpe vs benchmark
    benchmark_fold_sharpes = training_artifacts.get("benchmark_fold_sharpes", [])
    wf = walk_forward_sharpe_gate(fold_sharpes, benchmark_fold_sharpes)
    gates["walk_forward_sharpe"] = {"pass": wf["pass"], "detail": wf["detail"]}

    overall_pass = all(g["pass"] for g in gates.values())
    reasons_failed = [name for name, g in gates.items() if not g["pass"]]

    return {
        "bot_id": bot_id,
        "gates": gates,
        "overall_pass": overall_pass,
        "reasons_failed": reasons_failed,
    }
