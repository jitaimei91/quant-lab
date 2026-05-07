# src/quant_lab/backtest/harness.py
"""Walk-forward backtest harness.

Replays historical bars through the existing paper engine. For each window:
1. Initialize a fresh portfolio per strategy with `starting_cash`.
2. Step through each trading day in [window.train_end, window.test_end).
3. Slice histories so strategies only see data <= as_of (no leakage).
4. Run `run_morning_for_strategies` for one day, recording NAV.
5. Repeat next day.

Strategies that need refit/retrain on the train window must implement
`Strategy.fit(train_histories)` (Phase 3 ML); rule-based strategies are
parameter-free and a no-op on fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..tournament.runner import _avg_dollar_volume, run_morning_for_strategies
from ..strategies.base import Strategy
from ..types import Bar, Portfolio
from .windows import Window


@dataclass
class WalkForwardResult:
    nav_by_window: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    dates_by_window: dict[str, list[date]] = field(default_factory=dict)
    returns_by_window: dict[str, dict[str, list[float]]] = field(default_factory=dict)


def _slice_histories_to(
    histories: dict[str, list[Bar]],
    end_date: date,
) -> dict[str, list[Bar]]:
    return {sym: [b for b in bars if b.date <= end_date] for sym, bars in histories.items()}


def _trading_dates(histories: dict[str, list[Bar]], start: date, end: date) -> list[date]:
    dates: set[date] = set()
    for bars in histories.values():
        for b in bars:
            if start <= b.date < end:
                dates.add(b.date)
    return sorted(dates)


def run_walk_forward(
    strategies: list[Strategy],
    histories: dict[str, list[Bar]],
    windows: list[Window],
    starting_cash: float = 100_000.0,
    fit_callback=None,
) -> WalkForwardResult:
    """Run a walk-forward backtest over `windows`.

    `fit_callback(strategy, train_histories)` is called once per strategy at
    the start of each window with bars sliced to `[train_start, train_end)`.
    For Phase 1.5 this is a no-op; Phase 3 ML strategies will use it.
    """
    result = WalkForwardResult()

    for window in windows:
        # 1. Optional fit
        train_hist = {
            sym: [b for b in bars if window.train_start <= b.date < window.train_end]
            for sym, bars in histories.items()
        }
        if fit_callback is not None:
            for strat in strategies:
                fit_callback(strat, train_hist)

        # 2. Initialize fresh portfolios for this window
        portfolios: dict[str, Portfolio] = {
            strat.bot_id: Portfolio(bot_id=strat.bot_id, cash=starting_cash, positions={})
            for strat in strategies
        }
        navs: dict[str, list[tuple[date, float]]] = {strat.bot_id: [] for strat in strategies}

        # 3. Walk forward day by day
        for as_of in _trading_dates(histories, window.train_end, window.test_end):
            visible = _slice_histories_to(histories, as_of)
            advs = {sym: _avg_dollar_volume(bars) for sym, bars in visible.items()}
            portfolios, _trades, navs = run_morning_for_strategies(
                strategies=strategies,
                histories=visible,
                advs=advs,
                prior_portfolios=portfolios,
                prior_navs=navs,
                as_of=as_of,
                starting_cash=starting_cash,
            )

        # 4. Record results for this window
        result.nav_by_window[window.label] = {
            bot_id: [nav for _, nav in series] for bot_id, series in navs.items()
        }
        result.dates_by_window[window.label] = sorted(
            {d for series in navs.values() for d, _ in series}
        )
        result.returns_by_window[window.label] = {}
        for bot_id, series in navs.items():
            navs_only = [n for _, n in series]
            rets = [
                navs_only[i] / navs_only[i - 1] - 1.0
                for i in range(1, len(navs_only))
                if navs_only[i - 1] > 0
            ]
            result.returns_by_window[window.label][bot_id] = rets

    return result
