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


def _recent_trades_by_bot(
    trades_log_path: Path | None,
    limit_per_bot: int = 30,
) -> dict[str, list[dict]]:
    """Read trades.jsonl and bucket recent trades by bot_id."""
    if trades_log_path is None or not trades_log_path.exists():
        return {}
    by_bot: dict[str, list[dict]] = {}
    with trades_log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            by_bot.setdefault(row["bot_id"], []).append(row)
    return {bot: rows[-limit_per_bot:] for bot, rows in by_bot.items()}


def write_per_bot_files(
    out_dir: Path,
    leaderboard: list[tuple[str, Metrics, dict[str, float]]],
    nav_history: dict[str, list[tuple[date, float]]],
    trades_log_path: Path | None = None,
) -> None:
    """Write one JSON file per bot under out_dir/bots/<bot_id>.json.

    If `trades_log_path` is provided and exists, embeds the most recent trades
    for each bot directly in the per-bot JSON.
    """
    bots_dir = out_dir / "bots"
    bots_dir.mkdir(parents=True, exist_ok=True)
    trades_by_bot = _recent_trades_by_bot(trades_log_path)

    for bot_id, m, weights in leaderboard:
        nav_series = nav_history.get(bot_id, [])
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
            "recent_trades": trades_by_bot.get(bot_id, []),
        }
        (bots_dir / f"{bot_id}.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_validation_data(
    out_dir: Path,
    backtest_results_path: Path,
    live_metrics: dict[str, Metrics] | None = None,
) -> None:
    """Write data/validation.json from backtest_results.json plus optional live metrics.

    `live_metrics` lets the validation page show live-tournament Sharpe + alpha
    alongside the backtest aggregates.
    """
    if not backtest_results_path.exists():
        return

    raw = json.loads(backtest_results_path.read_text())
    strategies = raw.get("strategies", [])

    validation_entries = []
    for s in strategies:
        agg = s.get("aggregate", {})
        sig_weight = agg.get("significance_weight", 0.0)
        if sig_weight >= 0.7:
            badge = "green"
        elif sig_weight >= 0.3:
            badge = "yellow"
        else:
            badge = "gray"

        live_payload = None
        if live_metrics is not None:
            live = live_metrics.get(s["bot_id"])
            if live is not None:
                live_payload = {
                    "sharpe": live.sharpe,
                    "sharpe_ci_lo": live.sharpe_ci_lo,
                    "sharpe_ci_hi": live.sharpe_ci_hi,
                    "alpha_t_stat_vs_spy": live.alpha_t_stat_vs_spy,
                    "significance_weight": live.significance_weight,
                    "days": live.days,
                }

        validation_entries.append(
            {
                "bot_id": s["bot_id"],
                "aggregate": agg,
                "per_window": s.get("per_window", []),
                "significance_badge": badge,
                "failed_validation": sig_weight < 0.3,
                "live": live_payload,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps({"strategies": validation_entries}, indent=2) + "\n"
    )
