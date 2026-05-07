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


def write_dashboard_data(
    out_dir: Path,
    leaderboard: Iterable[tuple[str, Metrics, dict[str, float]]],
    nav_history: dict[str, list[tuple[date, float]]],
    market: dict[str, dict[str, float]],
    generated_at: date,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_payload = {
        "generated_at": generated_at.isoformat(),
        "market": market,
        "bots": [
            {
                "bot_id": bot_id,
                "metrics": {
                    "total_return": m.total_return,
                    "annualized_return": m.annualized_return,
                    "sharpe": m.sharpe,
                    "volatility": m.volatility,
                    "max_drawdown": m.max_drawdown,
                    "days": m.days,
                },
                "current_weights": weights,
            }
            for bot_id, m, weights in leaderboard
        ],
    }
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard_payload, indent=2) + "\n")

    nav_payload = {
        bot_id: [{"date": d.isoformat(), "nav": nav} for d, nav in series]
        for bot_id, series in nav_history.items()
    }
    (out_dir / "nav_history.json").write_text(json.dumps(nav_payload, indent=2) + "\n")
