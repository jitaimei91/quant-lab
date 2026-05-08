"""Online weight updates from live NAV tournament evidence.

When live paper-trading NAV accumulates data, this module re-computes strategy
weights using live Sharpe (with bootstrap CI) and alpha t-stat vs SPY. The
result is blended with backtest-calibrated weights using a confidence ramp:
new strategies (few live days) stay close to backtest; mature strategies
(365+ live days) use full live evidence.

The result is written to live_weights.json for MetaEnsemble to prefer over
the backtest-calibrated weights.
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
from .blend import blend_weights
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
    full_confidence_days: int = 365,
) -> dict[str, float]:
    """Recompute strategy weights from live NAV data using confidence-weighted blend.

    For each bot, computes live Sharpe (with bootstrap CI) and alpha t-stat vs SPY
    when enough data is available (>= min_days). Then blends backtest and live
    weights using a confidence ramp: bots with 0 live days get pure backtest weight;
    bots with full_confidence_days+ get pure live weight. In between, a smooth ramp.

    Writes the result to weights_path (live_weights.json).

    Args:
        nav_history: {bot_id: [(date, nav), ...]} loaded from persistence.
        benchmark_returns: {bot_id: [float, ...]} daily SPY returns aligned per bot.
            If a bot_id key is missing, uses the "SPY" key as fallback.
        min_days: Minimum live NAV days required to compute live weights for a bot.
            Bots below this use backtest weight only (confidence blend still applies).
        weights_path: Where to write live_weights.json.
        backtest_weights_path: Where to read backtest_results.json for fallback.
        n_iter: Bootstrap iterations (use 200 in tests for speed, 1000+ in prod).
        full_confidence_days: Days of live history for full confidence in live weights.

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

    # Build calibration records and track days-of-live per bot
    live_calibration_records: list[dict] = []
    days_of_live_per_bot: dict[str, int] = {}

    for bot_id, nav_series in nav_history.items():
        if bot_id == "meta-ensemble":
            continue
        returns = _nav_to_returns(nav_series)
        days_of_live_per_bot[bot_id] = len(returns)

        if len(returns) < min_days:
            # Not enough live data to compute meaningful live metrics
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
                # median_alpha_t feeds the evidence-weighted ensemble formula.
                # Single live window, so median == point estimate.
                "median_alpha_t": t_stat,
                "significance_weight": sig_w,
            },
            "per_window": [{"sharpe": sharpe_pt}],
        })

    # Compute weights for live-data-rich bots
    raw_live_weights: dict[str, float] = {}
    if live_calibration_records:
        raw_live_weights = compute_strategy_weights(live_calibration_records)

    # Confidence-weighted blend: smooth ramp from backtest to live
    merged = blend_weights(
        backtest_weights=fallback_weights,
        live_weights=raw_live_weights,
        days_of_live_per_bot=days_of_live_per_bot,
        full_confidence_days=full_confidence_days,
    )

    # If blend produced nothing (no weights in either dict), use backtest as-is
    if not merged and fallback_weights:
        total = sum(fallback_weights.values())
        if total > 0:
            merged = {k: v / total for k, v in fallback_weights.items()}

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
