"""Online weight updates from live NAV tournament evidence.

When live paper-trading NAV accumulates >= min_days of data, this module
re-computes strategy weights using live Sharpe (with bootstrap CI) and
alpha t-stat vs SPY. The result is written to live_weights.json for
MetaEnsemble to prefer over the backtest-calibrated weights.

Strategies with < min_days fall back to their backtest-calibrated weight.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..backtest.stats import (
    block_bootstrap_sharpe_ci,
    alpha_t_stat_vs_benchmark,
    significance_weight,
)
from .weights import compute_strategy_weights


def _nav_to_returns(nav_series: list[tuple[date, float]]) -> list[float]:
    """Convert (date, nav) pairs to daily return series."""
    rets: list[float] = []
    for i in range(1, len(nav_series)):
        prev = nav_series[i - 1][1]
        curr = nav_series[i][1]
        if prev > 0:
            rets.append(curr / prev - 1.0)
    return rets


def update_weights_from_live(
    nav_history: dict[str, list[tuple[date, float]]],
    benchmark_returns: dict[str, list[float]],
    min_days: int = 60,
    weights_path: Path | None = None,
    backtest_weights_path: Path | None = None,
    n_iter: int = 1000,
) -> dict[str, float]:
    """Recompute strategy weights from live NAV data.

    For bots with >= min_days of live NAV:
        1. Compute live Sharpe with bootstrap CI
        2. Compute alpha t-stat vs SPY benchmark
        3. Derive significance_weight
    Bots with < min_days fall back to their backtest-calibrated weight.

    Writes the result to weights_path (live_weights.json).

    Args:
        nav_history: {bot_id: [(date, nav), ...]} loaded from persistence.
        benchmark_returns: {bot_id: [float, ...]} daily SPY returns aligned per bot.
            If a bot_id key is missing, uses the "SPY" key as fallback.
        min_days: Minimum live NAV days required to use live weights.
        weights_path: Where to write live_weights.json.
        backtest_weights_path: Where to read backtest_results.json for fallback.
        n_iter: Bootstrap iterations (use 200 in tests for speed, 1000+ in prod).

    Returns:
        dict[str, float] of {bot_id: weight}.
    """
    # Load backtest fallback weights
    fallback_weights: dict[str, float] = {}
    if backtest_weights_path and backtest_weights_path.exists():
        try:
            data = json.loads(backtest_weights_path.read_text(encoding="utf-8"))
            strategies_list = data.get("strategies", [])
            if strategies_list:
                fallback_weights = compute_strategy_weights(strategies_list)
        except Exception:
            pass

    spy_rets = benchmark_returns.get("SPY", [])

    # Build calibration records for live-data-rich bots
    live_calibration_records: list[dict] = []
    fallback_bots: list[str] = []

    for bot_id, nav_series in nav_history.items():
        if bot_id == "meta-ensemble":
            continue
        returns = _nav_to_returns(nav_series)
        if len(returns) < min_days:
            fallback_bots.append(bot_id)
            continue

        # Live Sharpe with bootstrap CI
        sharpe_pt, ci_lo, ci_hi = block_bootstrap_sharpe_ci(
            returns, n_iter=n_iter, seed=42
        )
        # Alpha vs SPY
        bench = benchmark_returns.get(bot_id, spy_rets)
        n_bench = min(len(returns), len(bench))
        if n_bench >= 30:
            _alpha, t_stat = alpha_t_stat_vs_benchmark(
                returns[:n_bench], bench[:n_bench]
            )
        else:
            t_stat = 0.0

        sig_w = significance_weight(t_stat)

        live_calibration_records.append({
            "bot_id": bot_id,
            "aggregate": {
                "sharpe": sharpe_pt,
                "sharpe_ci_lo": ci_lo,
                "sharpe_ci_hi": ci_hi,
                "significance_weight": sig_w,
            },
            "per_window": [{"sharpe": sharpe_pt}],
        })

    # Compute weights for live-data-rich bots
    live_weights: dict[str, float] = {}
    if live_calibration_records:
        live_weights = compute_strategy_weights(live_calibration_records)

    # Merge: live weights take precedence; fallback bots use backtest weights
    merged: dict[str, float] = {}
    for bot_id in fallback_bots:
        fb_w = fallback_weights.get(bot_id, 0.0)
        if fb_w > 0:
            merged[bot_id] = fb_w

    for bot_id, w in live_weights.items():
        merged[bot_id] = w

    # Re-normalize the merged weights to sum = 1.0
    total = sum(merged.values())
    if total > 0:
        merged = {k: v / total for k, v in merged.items()}

    # Write to file
    if weights_path is not None:
        try:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(
                json.dumps(merged, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            print(f"[warn] Could not write live_weights.json: {exc}")

    return merged
