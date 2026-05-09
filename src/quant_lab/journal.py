"""Daily per-bot journal — audit trail + self-calibration data feed.

Writes one row per bot per morning run to `state/bot_journal.jsonl`. Each
row captures:
  - date, bot_id, weights (current target), nav, daily_return
  - rolling 30/60/90-day Sharpe (from the bot's own NAV series)
  - daily_return vs SPY, beats_spy_today (bool)
  - automated tag: "good_day" / "bad_day" / "neutral"

The lifecycle module (lifecycle.py) already computes the trailing-90-day
significance gate and auto-pauses bots; this journal is the human-readable
sibling — easy to grep, easy to ingest into a dashboard, easy to audit.

A "good_day" is defined as: positive daily return AND beat SPY for the day.
A "bad_day" is: negative daily return AND lost ≥ 0.5% to SPY.
Everything else is "neutral". Intentionally simple — the rolling Sharpes
are the real signal.

Design choice: JSONL append-only (one line per bot per day). Never rewrite
prior entries. Easy to diff, easy to ship to a database later, easy to
recover by re-running missed dates.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import date
from pathlib import Path
from typing import Any


TRADING_DAYS_PER_YEAR = 252


def _annualised_sharpe(returns: list[float]) -> float | None:
    """Annualised Sharpe ratio. Returns None if < 2 returns or zero std."""
    if len(returns) < 2:
        return None
    mean = statistics.mean(returns)
    std = statistics.stdev(returns)
    if std <= 0:
        return None
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def _nav_returns(nav_series: list[tuple[date, float]], window: int) -> list[float]:
    """Last `window` daily returns from NAV. Returns [] if fewer than `window`
    valid returns are available — we'd rather report None than a misleading
    "30d Sharpe" computed from 4 days."""
    if len(nav_series) < window + 1:
        return []
    rets: list[float] = []
    for i in range(1, len(nav_series)):
        prev = nav_series[i - 1][1]
        if prev > 0:
            rets.append(nav_series[i][1] / prev - 1.0)
    if len(rets) < window:
        return []
    return rets[-window:]


def _classify_day(daily_return: float, spy_return: float) -> str:
    """Tag a day as good / bad / neutral based on absolute and relative perf."""
    relative = daily_return - spy_return
    if daily_return > 0 and relative > 0:
        return "good_day"
    if daily_return < 0 and relative < -0.005:  # lost >= 50bps to SPY on a down day
        return "bad_day"
    return "neutral"


def _latest_daily_return(nav_series: list[tuple[date, float]]) -> float:
    """Most recent day's return from the NAV series. Available whenever we
    have ≥2 NAVs (in contrast to rolling Sharpes which require a full window)."""
    if len(nav_series) < 2:
        return 0.0
    prev = nav_series[-2][1]
    if prev <= 0:
        return 0.0
    return nav_series[-1][1] / prev - 1.0


def journal_entry(
    *,
    bot_id: str,
    today: date,
    weights: dict[str, float],
    nav_series: list[tuple[date, float]],
    spy_returns: list[float],
) -> dict[str, Any]:
    """Build a single journal entry. Pure function — no I/O."""
    nav = nav_series[-1][1] if nav_series else 0.0

    rets_30 = _nav_returns(nav_series, 30)
    rets_60 = _nav_returns(nav_series, 60)
    rets_90 = _nav_returns(nav_series, 90)

    daily_return = _latest_daily_return(nav_series)
    spy_today = spy_returns[-1] if spy_returns else 0.0
    relative = daily_return - spy_today

    return {
        "date": today.isoformat(),
        "bot_id": bot_id,
        "weights": {sym: round(w, 6) for sym, w in weights.items()},
        "nav": round(nav, 4),
        "daily_return": round(daily_return, 6),
        "spy_daily_return": round(spy_today, 6),
        "vs_spy": round(relative, 6),
        "beats_spy": daily_return > spy_today,
        "rolling_sharpe_30d": round(s, 4) if (s := _annualised_sharpe(rets_30)) is not None else None,
        "rolling_sharpe_60d": round(s, 4) if (s := _annualised_sharpe(rets_60)) is not None else None,
        "rolling_sharpe_90d": round(s, 4) if (s := _annualised_sharpe(rets_90)) is not None else None,
        "tag": _classify_day(daily_return, spy_today),
        "n_navs": len(nav_series),
    }


def append_journal(
    journal_path: Path,
    entries: list[dict[str, Any]],
) -> None:
    """Append entries to the JSONL journal. Idempotent on (date, bot_id):
    if a row for the same (date, bot_id) already exists, skip the new one
    rather than duplicating. Keeps the journal clean across re-runs of the
    same morning workflow.
    """
    if not entries:
        return
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str]] = set()
    if journal_path.exists():
        # Walk existing rows once to build the seen-set.
        # Cheap: thousands of rows, not millions.
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d, b = row.get("date"), row.get("bot_id")
            if d and b:
                seen.add((d, b))

    with journal_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            key = (entry["date"], entry["bot_id"])
            if key in seen:
                continue
            f.write(json.dumps(entry) + "\n")
            seen.add(key)


def write_journal_summary(
    journal_path: Path,
    summary_path: Path,
    *,
    lookback_days: int = 60,
) -> None:
    """Read the JSONL journal and write a per-bot rollup to `summary_path`.

    Each entry: bot_id, last 60 days' good_day/bad_day/neutral counts,
    latest rolling Sharpes, average daily-return-vs-SPY, current pause status.
    Consumed by the dashboard.
    """
    if not journal_path.exists():
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"bots": {}}, indent=2) + "\n", encoding="utf-8")
        return

    bots: dict[str, dict[str, Any]] = {}
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        bot_id = row.get("bot_id")
        if not bot_id:
            continue
        b = bots.setdefault(bot_id, {
            "bot_id": bot_id,
            "all_entries": [],
        })
        b["all_entries"].append(row)

    summary_bots: dict[str, Any] = {}
    for bot_id, b in bots.items():
        # Sort by date and keep only the last `lookback_days`
        rows = sorted(b["all_entries"], key=lambda r: r["date"])
        recent = rows[-lookback_days:]
        good = sum(1 for r in recent if r.get("tag") == "good_day")
        bad = sum(1 for r in recent if r.get("tag") == "bad_day")
        neutral = sum(1 for r in recent if r.get("tag") == "neutral")
        avg_vs_spy = (
            sum(r.get("vs_spy", 0.0) for r in recent) / len(recent)
            if recent else 0.0
        )
        latest = rows[-1] if rows else {}
        summary_bots[bot_id] = {
            "bot_id": bot_id,
            "lookback_days": min(lookback_days, len(recent)),
            "good_days": good,
            "bad_days": bad,
            "neutral_days": neutral,
            "win_rate": round(good / max(1, len(recent)), 4),
            "avg_vs_spy": round(avg_vs_spy, 6),
            "latest_rolling_sharpe_30d": latest.get("rolling_sharpe_30d"),
            "latest_rolling_sharpe_60d": latest.get("rolling_sharpe_60d"),
            "latest_rolling_sharpe_90d": latest.get("rolling_sharpe_90d"),
            "latest_nav": latest.get("nav"),
            "latest_date": latest.get("date"),
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"bots": summary_bots}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
