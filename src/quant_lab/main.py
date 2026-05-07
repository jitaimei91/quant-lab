"""Entry points for Quant Lab.

`morning_command` runs one full morning step: fetch data, run all registered
strategies, persist state, write dashboard data, post Discord brief.
"""
from __future__ import annotations

import argparse
import os
from datetime import date, datetime as _dt
from pathlib import Path

from . import strategies  # noqa: F401  (registers strategies via import)
from .data import fetch_history
from .tournament.factors import factor_proxies_from_histories
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
from .backtest.harness import run_walk_forward
from .backtest.report import write_calibration_report
from .backtest.slippage_sweep import run_slippage_sweep as _run_slippage_sweep
from .backtest.windows import regime_stress_windows, walk_forward_windows


SYMBOLS_FOR_PHASE_1 = ["SPY", "QQQ", "IWM", "VTV", "VUG"]


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

    # Build benchmark return series for extended metrics
    def _bar_rets(sym: str) -> list[float]:
        bars_sym = histories.get(sym, [])
        rets = []
        for i in range(1, len(bars_sym)):
            prev = bars_sym[i - 1].close
            if prev > 0:
                rets.append(bars_sym[i].close / prev - 1.0)
        return rets

    spy_rets = _bar_rets("SPY")
    qqq_rets = _bar_rets("QQQ")
    factor_rets = factor_proxies_from_histories(histories)

    for strat in strategies_list:
        navs = nav_history.get(strat.bot_id, [])
        nav_values = [n for _, n in navs]
        metrics = compute_metrics(
            nav_values,
            daily_returns_vs_spy=spy_rets if spy_rets else None,
            daily_returns_vs_qqq=qqq_rets if qqq_rets else None,
            factor_returns=factor_rets if factor_rets else None,
        )
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


def _benchmark_returns(histories, windows, symbol="SPY"):
    out = {}
    bars = histories.get(symbol, [])
    for window in windows:
        in_window = [b for b in bars if window.train_end <= b.date < window.test_end]
        rets = []
        for i in range(1, len(in_window)):
            prev = in_window[i - 1].close
            if prev > 0:
                rets.append(in_window[i].close / prev - 1.0)
        out[window.label] = rets
    return out


def backtest_command(
    out_dir,
    start,
    end,
    train_years: int = 5,
    step_months: int = 12,
    enable_slippage_sweep: bool = True,
    enable_regime_stress: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["SPY", "QQQ"]
    lookback_days = (end - start).days + 365
    histories = {}
    for sym in symbols:
        bars = fetch_history(sym, lookback_days=lookback_days)
        if bars:
            histories[sym] = [b for b in bars if start <= b.date <= end]
    if not histories:
        raise RuntimeError("No historical data fetched.")

    strategies_list = get_all()
    windows = walk_forward_windows(start=start, end=end, train_years=train_years, step_months=step_months)
    if not windows:
        raise RuntimeError(f"No walk-forward windows generated from {start} to {end} with train_years={train_years}.")

    wf_result = run_walk_forward(strategies=strategies_list, histories=histories, windows=windows)
    bench_returns = _benchmark_returns(histories, windows)

    sweep = None
    if enable_slippage_sweep:
        sweep = _run_slippage_sweep(
            strategies=strategies_list, histories=histories, windows=windows[:1]
        )
    regime_results = {}
    if enable_regime_stress:
        regimes = regime_stress_windows()
        applicable = [w for w in regimes if w.test_end <= end and w.train_start >= start]
        if applicable:
            regime_results["stress"] = run_walk_forward(
                strategies=strategies_list, histories=histories, windows=applicable
            )

    write_calibration_report(
        out_dir=out_dir,
        wf_result=wf_result,
        benchmark_returns_by_window=bench_returns,
        slippage_sweep=sweep,
        regime_results=regime_results,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(prog="quant-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("morning", help="Run the morning tournament step")

    bt = sub.add_parser("backtest", help="Run the walk-forward backtest")
    bt.add_argument("--start", type=lambda s: _dt.fromisoformat(s).date(), default=date(2015, 1, 1))
    bt.add_argument("--end", type=lambda s: _dt.fromisoformat(s).date(), default=date.today())
    bt.add_argument("--train-years", type=int, default=5)
    bt.add_argument("--step-months", type=int, default=12)
    bt.add_argument("--no-slippage-sweep", action="store_true")
    bt.add_argument("--no-regime-stress", action="store_true")

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
    elif args.cmd == "backtest":
        repo_root = Path(__file__).resolve().parents[2]
        backtest_command(
            out_dir=repo_root / "dashboard" / "data" / "backtest",
            start=args.start,
            end=args.end,
            train_years=args.train_years,
            step_months=args.step_months,
            enable_slippage_sweep=not args.no_slippage_sweep,
            enable_regime_stress=not args.no_regime_stress,
        )


if __name__ == "__main__":
    cli()
