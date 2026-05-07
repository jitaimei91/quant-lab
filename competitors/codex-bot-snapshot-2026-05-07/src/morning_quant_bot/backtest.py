from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .features import annualized_return, max_drawdown, returns_from_closes, sharpe_ratio, stdev, TRADING_DAYS
from .models import BacktestMetrics, Bar, StrategyParams
from .strategy import required_history, target_weights


@dataclass
class BacktestResult:
    params: StrategyParams
    metrics: BacktestMetrics
    equity_curve: list[float]
    dates: list[date]


def prepare_date_indexes(bars_by_symbol: dict[str, list[Bar]]) -> dict[str, dict[date, int]]:
    return {
        symbol: {bar.date: index for index, bar in enumerate(bars)}
        for symbol, bars in bars_by_symbol.items()
    }


def common_dates(bars_by_symbol: dict[str, list[Bar]]) -> list[date]:
    sets = [{bar.date for bar in bars} for bars in bars_by_symbol.values() if bars]
    if not sets:
        return []
    dates = set.intersection(*sets)
    return sorted(dates)


def latest_prices(bars_by_symbol: dict[str, list[Bar]]) -> dict[str, float]:
    return {symbol: bars[-1].close for symbol, bars in bars_by_symbol.items() if bars}


def run_buy_and_hold_benchmark(
    bars: list[Bar],
    start_cash: float = 10_000.0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestMetrics:
    filtered = bars
    if start_date is not None:
        filtered = [bar for bar in filtered if bar.date >= start_date]
    if end_date is not None:
        filtered = [bar for bar in filtered if bar.date <= end_date]
    if len(filtered) < 2 or filtered[0].close <= 0:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, start_cash)
    shares = start_cash / filtered[0].close
    equity_curve = [shares * bar.close for bar in filtered]
    return _metrics(equity_curve, start_cash, trades=1, turnover=1.0)


def run_backtest(
    bars_by_symbol: dict[str, list[Bar]],
    params: StrategyParams,
    start_cash: float = 10_000.0,
    slippage_bps: float = 5.0,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BacktestResult:
    dates = common_dates(bars_by_symbol)
    if start_date is not None:
        dates = [item for item in dates if item >= start_date]
    if end_date is not None:
        dates = [item for item in dates if item <= end_date]
    if len(dates) < required_history(params) + 30:
        return _empty_result(params, start_cash)

    date_indexes = prepare_date_indexes(bars_by_symbol)
    cash = start_cash
    shares = {symbol: 0.0 for symbol in bars_by_symbol}
    equity_curve: list[float] = []
    used_dates: list[date] = []
    trades = 0
    turnover = 0.0
    slippage = slippage_bps / 10_000.0
    start_index = required_history(params)

    for date_index, current_date in enumerate(dates[start_index:], start=start_index):
        prices = _prices_on(bars_by_symbol, date_indexes, current_date)
        equity = cash + sum(shares[symbol] * prices[symbol] for symbol in shares)

        should_rebalance = (date_index - start_index) % params.rebalance_days == 0
        if should_rebalance and equity > 0:
            weights, _reasons = target_weights(
                bars_by_symbol,
                date_indexes,
                current_date,
                params,
            )
            target_values = {
                symbol: equity * weights.get(symbol, 0.0)
                for symbol in shares
            }

            for symbol in sorted(shares):
                price = prices[symbol]
                current_value = shares[symbol] * price
                delta_value = target_values[symbol] - current_value
                if abs(delta_value) < max(5.0, equity * 0.001):
                    continue
                fill_price = price * (1 + slippage if delta_value > 0 else 1 - slippage)
                delta_shares = delta_value / fill_price
                cash -= delta_shares * fill_price
                shares[symbol] += delta_shares
                trades += 1
                turnover += abs(delta_value) / max(equity, 1.0)

        equity = cash + sum(shares[symbol] * prices[symbol] for symbol in shares)
        equity_curve.append(equity)
        used_dates.append(current_date)

    metrics = _metrics(equity_curve, start_cash, trades, turnover)
    return BacktestResult(params=params, metrics=metrics, equity_curve=equity_curve, dates=used_dates)


def split_periods(dates: list[date]) -> tuple[date, date, date, date] | None:
    if len(dates) < 500:
        return None
    start = dates[0]
    train_end = dates[int(len(dates) * 0.60)]
    validation_end = dates[int(len(dates) * 0.80)]
    end = dates[-1]
    return start, train_end, validation_end, end


def _prices_on(
    bars_by_symbol: dict[str, list[Bar]],
    date_indexes: dict[str, dict[date, int]],
    current_date: date,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol, bars in bars_by_symbol.items():
        index = date_indexes[symbol][current_date]
        prices[symbol] = bars[index].close
    return prices


def _metrics(
    equity_curve: list[float],
    start_cash: float,
    trades: int,
    turnover: float,
) -> BacktestMetrics:
    if not equity_curve:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, trades, turnover, start_cash)
    daily_returns = returns_from_closes(equity_curve)
    volatility = stdev(daily_returns) * (TRADING_DAYS ** 0.5) if daily_returns else 0.0
    final_equity = equity_curve[-1]
    total_return = final_equity / start_cash - 1.0 if start_cash > 0 else 0.0
    return BacktestMetrics(
        total_return=total_return,
        cagr=annualized_return([start_cash] + equity_curve),
        sharpe=sharpe_ratio([start_cash] + equity_curve),
        volatility=volatility,
        max_drawdown=max_drawdown([start_cash] + equity_curve),
        trades=trades,
        turnover=turnover,
        final_equity=final_equity,
    )


def _empty_result(params: StrategyParams, start_cash: float) -> BacktestResult:
    metrics = BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, start_cash)
    return BacktestResult(params=params, metrics=metrics, equity_curve=[], dates=[])
