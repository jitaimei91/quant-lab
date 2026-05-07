"""Morning tournament runner.

Loads each strategy's prior portfolio (or initializes), gets target weights,
applies paper-trading rebalance, records new NAV, returns updated state.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable

from ..engine import rebalance
from ..strategies.base import Strategy
from ..types import Bar, Portfolio, Trade


def _avg_dollar_volume(bars: list[Bar], window: int = 30) -> float:
    if len(bars) < 2:
        return 0.0
    # Use trailing window excluding current day
    recent = bars[-(window + 1):-1]
    if not recent:
        return 0.0
    return sum(b.close * b.volume for b in recent) / len(recent)


def run_morning_for_strategies(
    strategies: Iterable[Strategy],
    histories: dict[str, list[Bar]],
    advs: dict[str, float] | None,
    prior_portfolios: dict[str, Portfolio],
    prior_navs: dict[str, list[tuple[date, float]]],
    as_of: date,
    starting_cash: float = 100_000.0,
) -> tuple[dict[str, Portfolio], list[Trade], dict[str, list[tuple[date, float]]]]:
    """Run one morning step for all strategies. Returns updated state."""
    if advs is None:
        advs = {sym: _avg_dollar_volume(bars) for sym, bars in histories.items()}

    prices = {sym: bars[-1].close for sym, bars in histories.items() if bars}

    new_portfolios: dict[str, Portfolio] = {}
    new_navs: dict[str, list[tuple[date, float]]] = {k: list(v) for k, v in prior_navs.items()}
    all_trades: list[Trade] = []

    for strat in strategies:
        portfolio = prior_portfolios.get(
            strat.bot_id,
            Portfolio(bot_id=strat.bot_id, cash=starting_cash, positions={}),
        )
        weights = strat.target_weights(histories, as_of)
        result = rebalance(portfolio, weights, prices, advs, as_of=as_of)
        new_portfolios[strat.bot_id] = result.portfolio
        all_trades.extend(result.trades)

        nav = result.portfolio.equity(prices)
        series = new_navs.setdefault(strat.bot_id, [])
        if not series or series[-1][0] != as_of:
            series.append((as_of, nav))
        else:
            series[-1] = (as_of, nav)

    return new_portfolios, all_trades, new_navs
