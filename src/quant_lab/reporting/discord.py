"""Discord webhook reporter for the morning brief.

Discord limits messages to 2000 chars; this module truncates and links
to the dashboard for full detail.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

import requests

from ..tournament.stats import Metrics


DISCORD_MAX_CHARS = 2000

DISCLAIMER = "Research tool. Not financial advice. Paper trading only."


def build_message(
    today: date,
    leaderboard: Iterable[tuple[str, Metrics, dict[str, float]]],
    market: dict[str, dict[str, float]],
    dashboard_url: str | None = None,
) -> str:
    lines = [
        f"**QUANT LAB — {today.isoformat()}**",
        f"_{DISCLAIMER}_",
        "",
        "**Market**",
    ]
    for sym in ("SPY", "QQQ"):
        info = market.get(sym, {})
        chg = info.get("change_pct", 0.0)
        ytd = info.get("ytd_pct", 0.0)
        arrow = "▲" if chg >= 0 else "▼"
        lines.append(f"  {sym}: {chg:+.2f}% {arrow}  YTD {ytd:+.2f}%")
    lines.append("")
    lines.append("**Tournament**")
    for bot_id, metrics, weights in leaderboard:
        positions = ", ".join(f"{s} {w:.0%}" for s, w in sorted(weights.items()) if w > 0.01) or "cash"
        lines.append(
            f"  {bot_id}: total {metrics.total_return:+.2%} | "
            f"sharpe {metrics.sharpe:.2f} | dd {metrics.max_drawdown:+.2%} | {positions}"
        )
    if dashboard_url:
        lines.append("")
        lines.append(f"Dashboard: {dashboard_url}")
    msg = "\n".join(lines)
    if len(msg) > DISCORD_MAX_CHARS:
        msg = msg[: DISCORD_MAX_CHARS - 50] + "\n…(truncated; see dashboard)"
    return msg


def post_to_discord(webhook_url: str, message: str) -> None:
    if len(message) > DISCORD_MAX_CHARS:
        message = message[: DISCORD_MAX_CHARS - 50] + "\n…(truncated)"
    response = requests.post(webhook_url, json={"content": message}, timeout=15)
    if response.status_code >= 400:
        raise RuntimeError(f"Discord webhook returned {response.status_code}: {response.text[:200]}")
