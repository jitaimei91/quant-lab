"""Dashboard JSON exporter.

Writes a small set of JSON files served by GitHub Pages and consumed by
`dashboard/app.js` to render the leaderboard and equity curves.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

from ..tournament.stats import Metrics


def _metrics_dict(m: Metrics) -> dict:
    return {
        "total_return": m.total_return,
        "annualized_return": m.annualized_return,
        "sharpe": m.sharpe,
        "sharpe_ci_lo": m.sharpe_ci_lo,
        "sharpe_ci_hi": m.sharpe_ci_hi,
        "volatility": m.volatility,
        "max_drawdown": m.max_drawdown,
        "days": m.days,
        "alpha_t_stat_vs_spy": m.alpha_t_stat_vs_spy,
        "alpha_t_stat_vs_qqq": m.alpha_t_stat_vs_qqq,
        "significance_weight": m.significance_weight,
        "factor_loadings": m.factor_loadings,
    }


def write_dashboard_data(
    out_dir: Path,
    leaderboard: Iterable[tuple[str, Metrics, dict[str, float]]],
    nav_history: dict[str, list[tuple[date, float]]],
    market: dict[str, dict[str, float]],
    generated_at: date,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_list = list(leaderboard)

    leaderboard_payload = {
        "generated_at": generated_at.isoformat(),
        "market": market,
        "bots": [
            {
                "bot_id": bot_id,
                "metrics": _metrics_dict(m),
                "current_weights": weights,
            }
            for bot_id, m, weights in leaderboard_list
        ],
    }
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard_payload, indent=2) + "\n")

    nav_payload = {
        bot_id: [{"date": d.isoformat(), "nav": nav} for d, nav in series]
        for bot_id, series in nav_history.items()
    }
    (out_dir / "nav_history.json").write_text(json.dumps(nav_payload, indent=2) + "\n")

    write_per_bot_files(out_dir, leaderboard_list, nav_history)


def write_per_bot_files(
    out_dir: Path,
    leaderboard: list[tuple[str, Metrics, dict[str, float]]],
    nav_history: dict[str, list[tuple[date, float]]],
) -> None:
    """Write one JSON file per bot under out_dir/bots/<bot_id>.json."""
    bots_dir = out_dir / "bots"
    bots_dir.mkdir(parents=True, exist_ok=True)

    for bot_id, m, weights in leaderboard:
        nav_series = nav_history.get(bot_id, [])
        # Compute daily returns from NAV for equity curve
        daily_returns = []
        for i in range(1, len(nav_series)):
            prev = nav_series[i - 1][1]
            curr = nav_series[i][1]
            if prev > 0:
                daily_returns.append(
                    {"date": nav_series[i][0].isoformat(), "ret": curr / prev - 1.0}
                )

        payload = {
            "bot_id": bot_id,
            "metrics": _metrics_dict(m),
            "current_weights": weights,
            "nav_series": [
                {"date": d.isoformat(), "nav": nav} for d, nav in nav_series
            ],
            "daily_returns": daily_returns,
        }
        (bots_dir / f"{bot_id}.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_validation_data(
    out_dir: Path,
    backtest_results_path: Path,
) -> None:
    """Write data/validation.json from backtest_results.json for the validation page."""
    if not backtest_results_path.exists():
        return

    raw = json.loads(backtest_results_path.read_text())
    strategies = raw.get("strategies", [])

    validation_entries = []
    for s in strategies:
        agg = s.get("aggregate", {})
        sig_weight = agg.get("significance_weight", 0.0)
        # Badge: green >= 0.7, yellow 0.3-0.7, gray < 0.3
        if sig_weight >= 0.7:
            badge = "green"
        elif sig_weight >= 0.3:
            badge = "yellow"
        else:
            badge = "gray"

        validation_entries.append(
            {
                "bot_id": s["bot_id"],
                "aggregate": agg,
                "per_window": s.get("per_window", []),
                "significance_badge": badge,
                "failed_validation": sig_weight < 0.3,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps({"strategies": validation_entries}, indent=2) + "\n"
    )
