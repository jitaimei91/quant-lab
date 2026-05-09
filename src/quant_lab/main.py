"""Entry points for Quant Lab.

`morning_command` runs one full morning step: fetch data, run all registered
strategies, persist state, write dashboard data, post Discord brief.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime as _dt, timezone
from pathlib import Path

from . import strategies  # noqa: F401  (registers strategies via import)
from .data import fetch_history, fetch_history_range
from .tournament.factors import factor_proxies_from_histories
from .engine.regime import regime_state, should_pause_bot
from .persistence import (
    save_portfolios,
    load_portfolios,
    save_nav_history,
    load_nav_history,
    append_trades,
)
from .ensemble.live_calibration import update_weights_from_live
from .lifecycle import evaluate_lifecycle, load_lifecycle_state, save_lifecycle_state
from .reporting.discord import build_message, post_to_discord
from .reporting.dashboard import (
    write_dashboard_data,
    write_per_bot_files,
    write_validation_data,
)
from .journal import journal_entry, append_journal, write_journal_summary
from .strategies.base import get_all
from .tournament.runner import run_morning_for_strategies
from .tournament.stats import compute_metrics
from .backtest.harness import run_walk_forward
from .backtest.report import write_calibration_report
from .backtest.slippage_sweep import run_slippage_sweep as _run_slippage_sweep
from .backtest.windows import regime_stress_windows, walk_forward_windows


SYMBOLS_FOR_PHASE_1 = [
    "SPY", "QQQ", "IWM", "VTV", "VUG",
    "TLT", "IEF", "GLD", "USO",
    "EFA", "EEM", "VNQ", "HYG", "VXX",
    # Leveraged + vol-carry sleeves used by the Apex strategy. Other bots
    # may also see these in `histories.items()` loops; the per-bot weight
    # caps in the rebalance engine prevent any single bot from over-leveraging.
    "SSO", "TMF", "UGL", "SVXY", "SHY",
    # SPDR sector universe for the sector-momo bot.
    "XLK", "XLY", "XLV", "XLF", "XLP", "XLE", "XLI", "XLU", "XLRE", "XLB", "XLC",
    # Credit ETFs for the credit-carry bot (HYG already above).
    "LQD",
]

# Index symbols fetched for regime/diagnostics only — never passed to strategies
# (they're not tradable, so we keep them out of the tradable universe).
REGIME_SYMBOLS = ["^VIX"]


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


def _write_last_morning(state_dir: Path, status: str, strategy_ids: list[str]) -> None:
    """Write state/last_morning.json for the watchdog workflow."""
    payload = {
        "timestamp": _dt.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "strategies": strategy_ids,
    }
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "last_morning.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception as exc:
        print(f"[warn] Could not write last_morning.json: {exc}")


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

    completed_strategy_ids: list[str] = []
    run_status = "failed"
    try:
        _morning_command_inner(
            state_dir=state_dir,
            dashboard_data_dir=dashboard_data_dir,
            snapshot_dir=snapshot_dir,
            discord_webhook=discord_webhook,
            dashboard_url=dashboard_url,
            completed_strategy_ids=completed_strategy_ids,
        )
        run_status = "success"
    except Exception:
        run_status = "partial" if completed_strategy_ids else "failed"
        raise
    finally:
        _write_last_morning(state_dir, run_status, completed_strategy_ids)


def _morning_command_inner(
    state_dir: Path,
    dashboard_data_dir: Path,
    snapshot_dir: Path,
    discord_webhook: str | None,
    dashboard_url: str | None,
    completed_strategy_ids: list[str],
) -> None:
    histories: dict[str, list] = {}
    for symbol in SYMBOLS_FOR_PHASE_1:
        bars = fetch_history(symbol, lookback_days=400)
        if bars:
            histories[symbol] = bars
    if not histories:
        raise RuntimeError("No data fetched; check network or yfinance status.")

    # Fetch index-only series (e.g. ^VIX) into a separate dict so the regime
    # engine can read them without exposing non-tradables to strategies.
    regime_histories: dict[str, list] = {}
    for symbol in REGIME_SYMBOLS:
        bars = fetch_history(symbol, lookback_days=400)
        if bars:
            regime_histories[symbol] = bars

    today = max(bars[-1].date for bars in histories.values())

    prior_portfolios_list = load_portfolios(state_dir / "portfolios.json")
    prior_portfolios = {p.bot_id: p for p in prior_portfolios_list}
    prior_navs = load_nav_history(state_dir / "nav_history.json")

    # Regime check (VIX kill-switch). Pass tradable histories + regime-only
    # symbols so HMM features (SPY/TLT/SHY) and VIX are both available.
    reg = regime_state({**histories, **regime_histories})
    print(f"[regime] VIX={reg['vix']:.1f} regime={reg['regime']}")

    strategies_list = get_all()
    portfolios, trades, nav_history = run_morning_for_strategies(
        strategies=strategies_list,
        histories=histories,
        advs=None,
        prior_portfolios=prior_portfolios,
        prior_navs=prior_navs,
        as_of=today,
        block_new_entries=reg["halt_new_entries"],
        liquidate_all=reg["liquidate_all"],
    )

    save_portfolios(portfolios.values(), state_dir / "portfolios.json")
    save_nav_history(nav_history, state_dir / "nav_history.json")
    append_trades(trades, state_dir / "trades.jsonl")

    # Live calibration: update ensemble weights from accumulated live NAV evidence.
    # Writes to live_weights.json; MetaEnsemble picks it up on next run.
    try:
        spy_rets_live = []
        spy_bars = histories.get("SPY", [])
        for i in range(1, len(spy_bars)):
            prev = spy_bars[i - 1].close
            if prev > 0:
                spy_rets_live.append(spy_bars[i].close / prev - 1.0)
        bench_rets = {"SPY": spy_rets_live}
        repo_root = Path(__file__).resolve().parents[2]
        live_weights_path = dashboard_data_dir / "backtest" / "live_weights.json"
        backtest_results_path = repo_root / "dashboard" / "data" / "backtest" / "backtest_results.json"
        update_weights_from_live(
            nav_history=nav_history,
            benchmark_returns=bench_rets,
            min_days=60,
            weights_path=live_weights_path,
            backtest_weights_path=backtest_results_path,
        )
    except Exception as exc:
        print(f"[warn] Live calibration update failed: {exc}")

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

    paused_bots: dict[str, str] = {}
    journal_entries: list[dict] = []
    for strat in strategies_list:
        navs = nav_history.get(strat.bot_id, [])
        nav_values = [n for _, n in navs]
        metrics = compute_metrics(
            nav_values,
            daily_returns_vs_spy=spy_rets if spy_rets else None,
            daily_returns_vs_qqq=qqq_rets if qqq_rets else None,
            factor_returns=factor_rets if factor_rets else None,
        )
        # Per-bot drawdown/Sharpe pause check
        paused, reason = should_pause_bot(strat.bot_id, navs)
        if paused:
            paused_bots[strat.bot_id] = reason
            print(f"[regime] {strat.bot_id} paused: {reason}")

        portfolio = portfolios[strat.bot_id]
        weights = {
            sym: portfolio.weight(sym, last_prices)
            for sym in portfolio.positions
        }
        leaderboard.append((strat.bot_id, metrics, weights))

        # Daily journal row — pure function, no side effects until we append
        try:
            journal_entries.append(
                journal_entry(
                    bot_id=strat.bot_id,
                    today=today,
                    weights=weights,
                    nav_series=navs,
                    spy_returns=spy_rets,
                )
            )
        except Exception as exc:
            print(f"[warn] journal entry failed for {strat.bot_id}: {exc}")

    # Persist daily journal (idempotent on (date, bot_id))
    try:
        journal_path = state_dir / "bot_journal.jsonl"
        append_journal(journal_path, journal_entries)
        write_journal_summary(
            journal_path=journal_path,
            summary_path=dashboard_data_dir / "journal_summary.json",
            lookback_days=60,
        )
    except Exception as exc:
        print(f"[warn] journal append failed: {exc}")

    # Lifecycle: auto-pause/resume based on rolling significance
    lifecycle_state: dict = {}
    try:
        lifecycle_path = state_dir / "strategy_lifecycle.json"
        prior_lifecycle = load_lifecycle_state(lifecycle_path)
        lifecycle_state = evaluate_lifecycle(
            nav_history=nav_history,
            benchmark_returns={"SPY": spy_rets},
            prior_state=prior_lifecycle,
            today=today,
        )
        save_lifecycle_state(lifecycle_state, lifecycle_path)
        # Write dashboard/data/lifecycle.json for the validation page
        import json as _json
        lifecycle_dashboard: dict[str, dict] = {}
        for bot_id, ls in lifecycle_state.items():
            lifecycle_dashboard[bot_id] = {
                "bot_id": ls.bot_id,
                "paused": ls.paused,
                "paused_at": ls.paused_at.isoformat() if ls.paused_at else None,
                "pause_reason": ls.pause_reason,
                "consecutive_fail_days": ls.consecutive_fail_days,
                "consecutive_recovery_days": ls.consecutive_recovery_days,
            }
        lifecycle_data_path = dashboard_data_dir / "lifecycle.json"
        lifecycle_data_path.write_text(
            _json.dumps(lifecycle_dashboard, indent=2) + "\n", encoding="utf-8"
        )
        for bot_id, ls in lifecycle_state.items():
            if ls.paused:
                print(f"[lifecycle] {bot_id} paused: {ls.pause_reason}")
    except Exception as exc:
        print(f"[warn] Lifecycle evaluation failed: {exc}")

    # Include lifecycle-paused bots in the sort-to-bottom set
    lifecycle_paused_bots: set[str] = {
        bot_id for bot_id, ls in lifecycle_state.items() if ls.paused
    }

    # Sort active (non-paused) bots first by Sharpe, then paused bots (regime or lifecycle)
    leaderboard.sort(
        key=lambda row: (
            row[0] in paused_bots or row[0] in lifecycle_paused_bots,
            -row[1].sharpe,
        )
    )

    # Record completed strategy IDs for watchdog last_morning.json
    completed_strategy_ids.extend([strat.bot_id for strat in strategies_list])

    market = _market_snapshot(histories, today)
    write_dashboard_data(
        out_dir=dashboard_data_dir,
        leaderboard=leaderboard,
        nav_history=nav_history,
        market=market,
        generated_at=today,
    )

    # Refresh per-bot JSONs with embedded recent trades from trades.jsonl
    write_per_bot_files(
        out_dir=dashboard_data_dir,
        leaderboard=leaderboard,
        nav_history=nav_history,
        trades_log_path=state_dir / "trades.jsonl",
    )

    # Write validation data, merging live metrics alongside backtest aggregates
    repo_root_for_bt = Path(__file__).resolve().parents[2]
    backtest_results_path = repo_root_for_bt / "dashboard" / "data" / "backtest" / "backtest_results.json"
    live_metrics_by_bot = {bot_id: m for bot_id, m, _ in leaderboard}
    try:
        write_validation_data(
            out_dir=dashboard_data_dir,
            backtest_results_path=backtest_results_path,
            live_metrics=live_metrics_by_bot,
            lifecycle_state=lifecycle_state if lifecycle_state else None,
        )
    except Exception as exc:
        print(f"[warn] write_validation_data failed: {exc}")

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
    universe_path: Path | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load universe: SPY+QQQ always, plus universe file if available
    from .data.universe import load_universe as _load_universe
    repo_root = Path(__file__).resolve().parents[2]
    if universe_path is None:
        universe_path = repo_root / "config" / "universe_r1000.txt"
    if universe_path.exists():
        symbols = _load_universe(universe_path)
    else:
        symbols = ["SPY", "QQQ"]
    # Fetch the full range needed: training data starts train_years before `start`
    from datetime import timedelta as _td
    fetch_start = date(max(start.year - train_years - 1, 2000), 1, 1)
    histories = {}
    print(f"[backtest] Fetching {len(symbols)} symbols from {fetch_start} to {end} ...")
    for sym in symbols:
        bars = fetch_history_range(sym, start=fetch_start, end=end)
        if bars:
            histories[sym] = bars
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


def ml_train_command(
    state_dir: Path,
    dashboard_data_dir: Path,
    models_dir: Path,
    start: date,
    end: date,
    horizon: int = 5,
    seed: int = 42,
    universe_path: Path | None = None,
) -> None:
    """Walk-forward ML training pipeline: train, validate, persist state."""
    from .data.universe import load_universe as _load_universe
    from .ml.train import train_xgboost_walkforward, train_lightgbm_walkforward
    from .ml.validate import label_shuffle_test, run_all_gates
    from .ml.features import build_training_set

    state_dir.mkdir(parents=True, exist_ok=True)
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    if universe_path is None:
        universe_path = repo_root / "config" / "universe_r1000.txt"
    if universe_path.exists():
        symbols = _load_universe(universe_path)
    else:
        symbols = ["SPY", "QQQ"]

    lookback_days = (end - start).days + 365
    print(f"[ml-train] Fetching {len(symbols)} symbols ...")
    histories: dict = {}
    for sym in symbols:
        bars = fetch_history(sym, lookback_days=lookback_days)
        if bars:
            histories[sym] = bars
    if not histories:
        raise RuntimeError("No data fetched for ML training.")

    target_symbols = [s for s in histories if s not in {"SPY", "QQQ", "^VIX"}]
    windows = walk_forward_windows(start=start, end=end, train_years=3, step_months=12)
    if not windows:
        raise RuntimeError("No walk-forward windows generated.")

    print(f"[ml-train] Training with {len(windows)} walk-forward windows ...")

    # --- Build full training set for label-shuffle test ---
    X_full, y_full = build_training_set(
        histories=histories,
        target_symbols=target_symbols,
        train_start=windows[0].train_start,
        train_end=windows[-1].train_end,
        horizon=horizon,
        sample_every_days=5,
    )

    # Benchmark fold Sharpes (SPY buy-and-hold annualised daily return)
    bench_sharpes: list[float] = []
    import math as _math

    import numpy as np
    spy_bars = histories.get("SPY", [])
    for w in windows:
        spy_in_w = [b for b in spy_bars if w.test_start <= b.date < w.test_end]
        if len(spy_in_w) > 1:
            rets = [spy_in_w[i].close / spy_in_w[i - 1].close - 1.0 for i in range(1, len(spy_in_w))]
            arr = np.array(rets)
            std = float(np.std(arr, ddof=1))
            bench_sharpes.append(float(np.mean(arr)) / std * _math.sqrt(252) if std > 0 else 0.0)

    ml_validation: dict = {}

    for train_fn_name, train_fn, bot_id in [
        ("xgboost", train_xgboost_walkforward, "gradboost"),
        ("lightgbm", train_lightgbm_walkforward, "lightforest"),
    ]:
        print(f"[ml-train] Training {bot_id} ...")
        artifacts = train_fn(
            histories=histories,
            target_symbols=target_symbols,
            windows=windows,
            horizon=horizon,
            seed=seed,
            models_dir=models_dir,
        )

        # Label-shuffle test
        if not X_full.empty:
            def _simple_train(X, y):
                from sklearn.linear_model import Ridge
                m = Ridge()
                m.fit(X.fillna(0.0).values, y.values)
                return m

            shuffle_result = label_shuffle_test(
                train_fn=_simple_train,
                X=X_full,
                y=y_full,
                n_shuffles=10,
                seed=seed,
            )
        else:
            shuffle_result = {"pass": True, "detail": "No training data for shuffle test"}

        gate_artifacts = {
            "label_shuffle_result": shuffle_result,
            "fold_sharpes": artifacts["fold_sharpes"],
            "benchmark_fold_sharpes": bench_sharpes,
            "live_sharpe": None,
        }
        gate_result = run_all_gates(bot_id, gate_artifacts)
        ml_validation[bot_id] = gate_result
        status = "PASS" if gate_result["overall_pass"] else f"FAIL: {gate_result['reasons_failed']}"
        print(f"[ml-train] {bot_id} gates: {status}")

    # Ensemble: passes iff both components pass
    both_pass = all(ml_validation.get(bid, {}).get("overall_pass", False) for bid in ["gradboost", "lightforest"])
    ml_validation["ml-ensemble"] = {
        "bot_id": "ml-ensemble",
        "overall_pass": both_pass,
        "reasons_failed": [] if both_pass else ["component bots failed gates"],
        "gates": {},
    }

    # Write validation state
    validation_path = state_dir / "ml_validation.json"
    validation_path.write_text(json.dumps(ml_validation, indent=2) + "\n")
    print(f"[ml-train] Validation state written to {validation_path}")

    # Write failures to dashboard
    failures = {bid: v for bid, v in ml_validation.items() if not v.get("overall_pass", False)}
    if failures:
        failed_path = dashboard_data_dir / "validation_failed.json"
        failed_path.write_text(json.dumps(failures, indent=2) + "\n")
        print(f"[ml-train] {len(failures)} bots failed gates → {failed_path}")
    else:
        print("[ml-train] All ML bots passed gates.")


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
    bt.add_argument("--universe-file", type=Path, default=None)

    mlt = sub.add_parser("ml-train", help="Walk-forward ML training + validation gates")
    mlt.add_argument("--start", type=lambda s: _dt.fromisoformat(s).date(), default=date(2020, 1, 1))
    mlt.add_argument("--end", type=lambda s: _dt.fromisoformat(s).date(), default=date.today())
    mlt.add_argument("--horizon", type=int, default=5)
    mlt.add_argument("--seed", type=int, default=42)
    mlt.add_argument("--universe-file", type=Path, default=None)

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
            universe_path=args.universe_file,
        )
    elif args.cmd == "ml-train":
        repo_root = Path(__file__).resolve().parents[2]
        ml_train_command(
            state_dir=repo_root / "state",
            dashboard_data_dir=repo_root / "dashboard" / "data",
            models_dir=repo_root / "models",
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            seed=args.seed,
            universe_path=args.universe_file,
        )


if __name__ == "__main__":
    cli()
