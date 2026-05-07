from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from .backtest import latest_prices, prepare_date_indexes, run_backtest
from .config import load_account, read_universe, save_account
from .data import MarketDataClient, fetch_histories
from .evolver import StrategyEvolver
from .export import rebuild_public_site, write_public_artifacts
from .models import Account, Recommendation
from .paper import apply_paper_fills
from .paths import (
    CACHE_DIR,
    DEFAULT_ACCOUNT_PATH,
    DEFAULT_UNIVERSE_PATH,
    LEADERBOARD_PATH,
    PAPER_ACCOUNT_PATH,
    PUBLIC_DIR,
    REPORTS_DIR,
    RUN_LOG_PATH,
    TRADE_LOG_PATH,
)
from .report import build_report
from .strategy import target_weights


def run_morning(
    account_path: Path = DEFAULT_ACCOUNT_PATH,
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
    years: int = 8,
    generations: int = 4,
    population: int = 32,
    paper_fill: bool = False,
    bot_id: str = "morning-quant-v1",
    redact_public: bool = False,
) -> Path:
    account = _load_account_for_morning(account_path)
    universe = read_universe(universe_path)
    symbols = sorted(set(universe) | set(account.positions))

    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    client = MarketDataClient(CACHE_DIR)
    histories = fetch_histories(client, symbols, start, end, min_rows=_min_rows(years))

    evolver = StrategyEvolver(LEADERBOARD_PATH)
    records = evolver.evolve(histories, generations=generations, population_size=population)
    champion = records[0]

    prices = latest_prices(histories)
    equity = account.equity(prices)
    latest_date = max(bar.date for bars in histories.values() for bar in bars)
    date_indexes = prepare_date_indexes(histories)
    weights, reasons = target_weights(histories, date_indexes, latest_date, champion.params)
    recommendations = _recommendations(account, equity, prices, weights, reasons)

    report_path = REPORTS_DIR / f"morning-report-{end.isoformat()}.md"
    build_report(
        report_date=end,
        account=account,
        equity=equity,
        strategy=champion.params,
        metrics=champion.test,
        recommendations=recommendations,
        report_path=report_path,
        data_start=start,
        data_end=latest_date,
    )

    write_public_artifacts(
        PUBLIC_DIR,
        RUN_LOG_PATH,
        bot_id=bot_id,
        report_date=end,
        report_path=report_path,
        account=account,
        equity=equity,
        data_start=start,
        data_end=latest_date,
        strategy=champion.params,
        metrics=champion.test,
        recommendations=recommendations,
        redact=redact_public,
    )

    if paper_fill:
        updated = apply_paper_fills(account, recommendations, TRADE_LOG_PATH)
        save_account(PAPER_ACCOUNT_PATH, updated)

    return report_path


def run_evolution_only(
    universe_path: Path = DEFAULT_UNIVERSE_PATH,
    years: int = 8,
    generations: int = 6,
    population: int = 48,
) -> int:
    universe = read_universe(universe_path)
    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    histories = fetch_histories(
        MarketDataClient(CACHE_DIR),
        universe,
        start,
        end,
        min_rows=_min_rows(years),
    )
    records = StrategyEvolver(LEADERBOARD_PATH).evolve(
        histories,
        generations=generations,
        population_size=population,
    )
    return len(records)


def run_simple_backtest(
    symbols: list[str],
    years: int = 8,
    generations: int = 3,
    population: int = 24,
) -> tuple[Path, float]:
    end = date.today()
    start = end - timedelta(days=int(years * 365.25))
    histories = fetch_histories(
        MarketDataClient(CACHE_DIR),
        symbols,
        start,
        end,
        min_rows=_min_rows(years),
    )
    records = StrategyEvolver(LEADERBOARD_PATH).evolve(
        histories,
        generations=generations,
        population_size=population,
    )
    result = run_backtest(histories, records[0].params)
    path = REPORTS_DIR / f"backtest-{end.isoformat()}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics
    path.write_text(
        "\n".join(
            [
                f"Symbols: {', '.join(symbols)}",
                f"Total return: {metrics.total_return:.2%}",
                f"CAGR: {metrics.cagr:.2%}",
                f"Sharpe: {metrics.sharpe:.2f}",
                f"Max drawdown: {metrics.max_drawdown:.2%}",
                f"Trades: {metrics.trades}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path, metrics.sharpe


def export_site() -> Path:
    rebuild_public_site(PUBLIC_DIR, RUN_LOG_PATH)
    return PUBLIC_DIR / "index.html"


def _load_account_for_morning(account_path: Path) -> Account:
    if account_path.exists():
        return load_account(account_path)
    if PAPER_ACCOUNT_PATH.exists():
        return load_account(PAPER_ACCOUNT_PATH)
    return load_account(account_path)


def _min_rows(years: int) -> int:
    return max(80, min(260, int(years * 210)))


def _recommendations(
    account: Account,
    equity: float,
    prices: dict[str, float],
    weights: dict[str, float],
    reasons: dict[str, str],
) -> list[Recommendation]:
    symbols = sorted(set(account.positions) | set(weights))
    recs: list[Recommendation] = []
    for symbol in symbols:
        latest_price = prices.get(symbol)
        if latest_price is None or latest_price <= 0:
            continue
        current_shares = account.positions.get(symbol).shares if symbol in account.positions else 0.0
        target_weight = weights.get(symbol, 0.0)
        target_value = equity * target_weight
        target_shares = target_value / latest_price
        delta_shares = target_shares - current_shares
        threshold = max(0.01, abs(target_shares) * 0.02)
        if abs(delta_shares) <= threshold:
            action = "HOLD"
        elif delta_shares > 0:
            action = "BUY"
        else:
            action = "SELL"
        if action == "HOLD" and symbol not in weights:
            action = "SELL" if current_shares > 0 else "HOLD"
        reason = reasons.get(symbol, "risk filter or target rebalance")
        recs.append(
            Recommendation(
                symbol=symbol,
                action=action,
                current_shares=current_shares,
                target_shares=target_shares,
                delta_shares=delta_shares,
                latest_price=latest_price,
                target_weight=target_weight,
                reason=reason,
            )
        )
    recs.sort(key=lambda item: (item.action == "HOLD", item.symbol))
    return recs
