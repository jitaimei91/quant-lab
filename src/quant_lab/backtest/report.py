# src/quant_lab/backtest/report.py
"""Aggregate walk-forward + slippage-sweep + regime-stress results into a
calibration report (JSON for the dashboard, Markdown for human reading).
"""
from __future__ import annotations

import json
from pathlib import Path

from .harness import WalkForwardResult
from .stats import (
    alpha_t_stat_vs_benchmark,
    block_bootstrap_sharpe_ci,
    significance_weight,
)


def _per_strategy_summary(
    wf_result: WalkForwardResult,
    benchmark_returns_by_window: dict[str, list[float]],
) -> list[dict]:
    bot_ids = {b for w in wf_result.returns_by_window.values() for b in w.keys()}
    summaries = []
    for bot_id in sorted(bot_ids):
        per_window = []
        all_returns: list[float] = []
        all_alpha_ts: list[float] = []
        for window_label, by_bot in wf_result.returns_by_window.items():
            rets = by_bot.get(bot_id, [])
            if not rets:
                continue
            point, lo, hi = block_bootstrap_sharpe_ci(rets, n_iter=500, block_len=20, seed=42)
            bench = benchmark_returns_by_window.get(window_label, [])
            alpha, t = alpha_t_stat_vs_benchmark(rets, bench) if bench else (0.0, 0.0)
            per_window.append({
                "window": window_label,
                "sharpe": point,
                "sharpe_ci_lo": lo,
                "sharpe_ci_hi": hi,
                "alpha_per_day": alpha,
                "alpha_t_stat": t,
                "days": len(rets),
            })
            all_returns.extend(rets)
            all_alpha_ts.append(t)
        if not per_window:
            continue
        agg_point, agg_lo, agg_hi = block_bootstrap_sharpe_ci(all_returns, n_iter=1000, block_len=20, seed=42)
        median_t = sorted(all_alpha_ts)[len(all_alpha_ts) // 2] if all_alpha_ts else 0.0
        sig_weight = significance_weight(median_t)
        summaries.append({
            "bot_id": bot_id,
            "aggregate": {
                "sharpe": agg_point,
                "sharpe_ci_lo": agg_lo,
                "sharpe_ci_hi": agg_hi,
                "median_alpha_t": median_t,
                "significance_weight": sig_weight,
                "windows_evaluated": len(per_window),
                "total_test_days": len(all_returns),
            },
            "per_window": per_window,
        })
    return summaries


def write_calibration_report(
    out_dir: Path,
    wf_result: WalkForwardResult,
    benchmark_returns_by_window: dict[str, list[float]],
    slippage_sweep,
    regime_results: dict[str, WalkForwardResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = _per_strategy_summary(wf_result, benchmark_returns_by_window)

    payload = {
        "strategies": summaries,
        "regimes": {
            label: _per_strategy_summary(result, {})
            for label, result in regime_results.items()
        },
        "slippage_sweep": (
            {
                str(mult): _per_strategy_summary(res, benchmark_returns_by_window)
                for mult, res in slippage_sweep.results.items()
            }
            if slippage_sweep is not None
            else None
        ),
    }
    (out_dir / "backtest_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "backtest_curves.json").write_text(
        json.dumps({
            "windows": list(wf_result.nav_by_window.keys()),
            "curves": {
                label: {
                    bot_id: [
                        {"date": d.isoformat(), "nav": nav}
                        for d, nav in zip(wf_result.dates_by_window[label], navs)
                    ]
                    for bot_id, navs in wf_result.nav_by_window[label].items()
                }
                for label in wf_result.nav_by_window
            },
        }, indent=2) + "\n"
    )

    # Markdown report
    lines = ["# Calibration Report", ""]
    lines.append(f"Strategies evaluated: **{len(summaries)}**")
    lines.append("")
    lines.append("| Bot | Aggregate Sharpe | 95% CI | Median α t-stat | Sig weight | Days |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        a = s["aggregate"]
        lines.append(
            f"| {s['bot_id']} | {a['sharpe']:.2f} | [{a['sharpe_ci_lo']:.2f}, {a['sharpe_ci_hi']:.2f}] | "
            f"{a['median_alpha_t']:.2f} | {a['significance_weight']:.2f} | {a['total_test_days']} |"
        )
    (out_dir / "calibration_report.md").write_text("\n".join(lines) + "\n")
