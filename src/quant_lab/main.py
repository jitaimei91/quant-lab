"""Entry points for Quant Lab.

`morning_command` runs one full morning step: fetch data, run all registered
strategies, persist state, write dashboard data, post Discord brief.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from . import strategies  # noqa: F401  (registers strategies via import)
from .data import fetch_history
from .persistence import (
    save_portfolios,
    load_portfolios,
    save_nav_history,
    load_nav_history,
    append_trades,
)
from .reporting.discord import build_message, post_to_discord
from .reporting.dashboard import write_dashboard_data
from .strategies.base import get_all
from .tournament.runner import run_morning_for_strategies
from .tournament.stats import compute_metrics


SYMBOLS_FOR_PHASE_1 = ["SPY", "QQQ"]


def _market_snapshot(histories: dict, today: date) -> dict[str, dict[str, float]]:
    snapshot: dict[str, dict[str, float]] = {}
    for sym in ("SPY", "QQQ"):
        bars = histories.get(sym, [])
        if len(bars) < 2:
            snapshot[sym] = {"change_pct": 0.0, "ytd_pct": 0.0}
            continue
        last = bars[-1]
        prev = bars[-2]
        chg = (last.close / prev.close - 1.0) * 100
        ytd = next((b for b in bars if b.date.year == last.date.year), bars[0])
        ytd_pct = (last.close / ytd.close - 1.0) * 100
        snapshot[sym] = {"change_pct": chg, "ytd_pct": ytd_pct}
    return snapshot


def morning_command(
    state_dir: Path,
    dashboard_data_dir: Path,
    snapshot_dir: Path,
    discord_webhook: str | None,
    dashboard_url: str | None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, list] = {}
    for symbol in SYMBOLS_FOR_PHASE_1:
        bars = fetch_history(symbol, lookback_days=400)
        if bars:
            histories[symbol] = bars
    if not histories:
        raise RuntimeError("No data fetched; check network or yfinance status.")

    today = max(bars[-1].date for bars in histories.values())

    prior_portfolios_list = load_portfolios(state_dir / "portfolios.json")
    prior_portfolios = {p.bot_id: p for p in prior_portfolios_list}
    prior_navs = load_nav_history(state_dir / "nav_history.json")

    strategies_list = get_all()
    portfolios, trades, nav_history = run_morning_for_strategies(
        strategies=strategies_list,
        histories=histories,
        advs=None,
        prior_portfolios=prior_portfolios,
        prior_navs=prior_navs,
        as_of=today,
    )

    save_portfolios(portfolios.values(), state_dir / "portfolios.json")
    save_nav_history(nav_history, state_dir / "nav_history.json")
    append_trades(trades, state_dir / "trades.jsonl")

    leaderboard = []
    last_prices = {s: bars[-1].close for s, bars in histories.items()}
    for strat in strategies_list:
        navs = nav_history.get(strat.bot_id, [])
        nav_values = [n for _, n in navs]
        metrics = compute_metrics(nav_values)
        portfolio = portfolios[strat.bot_id]
        weights = {
            sym: portfolio.weight(sym, last_prices)
            for sym in portfolio.positions
        }
        leaderboard.append((strat.bot_id, metrics, weights))
    leaderboard.sort(key=lambda row: row[1].sharpe, reverse=True)

    market = _market_snapshot(histories, today)
    write_dashboard_data(
        out_dir=dashboard_data_dir,
        leaderboard=leaderboard,
        nav_history=nav_history,
        market=market,
        generated_at=today,
    )

    if discord_webhook:
        msg = build_message(today, leaderboard, market, dashboard_url=dashboard_url)
        try:
            post_to_discord(discord_webhook, msg)
        except Exception as exc:
            # Log but don't crash the run; dashboard still updates
            print(f"[warn] Discord post failed: {exc}")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="quant-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("morning", help="Run the morning tournament step")

    args = parser.parse_args()
    if args.cmd == "morning":
        repo_root = Path(__file__).resolve().parents[2]
        morning_command(
            state_dir=repo_root / "state",
            dashboard_data_dir=repo_root / "dashboard" / "data",
            snapshot_dir=repo_root / "data" / "snapshots",
            discord_webhook=os.getenv("DISCORD_WEBHOOK"),
            dashboard_url=os.getenv("DASHBOARD_URL"),
        )


if __name__ == "__main__":
    cli()
