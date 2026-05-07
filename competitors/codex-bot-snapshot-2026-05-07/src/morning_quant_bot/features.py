from __future__ import annotations

from math import sqrt

TRADING_DAYS = 252


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return sqrt(max(variance, 0.0))


def returns_from_closes(closes: list[float]) -> list[float]:
    output: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous > 0:
            output.append(current / previous - 1.0)
    return output


def sma(values: list[float], period: int, index: int) -> float | None:
    if index + 1 < period:
        return None
    window = values[index - period + 1 : index + 1]
    return mean(window)


def annualized_volatility(closes: list[float], period: int, index: int) -> float:
    if index < period:
        return 0.0
    window = closes[index - period : index + 1]
    daily_returns = returns_from_closes(window)
    return stdev(daily_returns) * sqrt(TRADING_DAYS)


def sharpe_ratio(equity_curve: list[float]) -> float:
    daily_returns = returns_from_closes(equity_curve)
    if len(daily_returns) < 2:
        return 0.0
    volatility = stdev(daily_returns)
    if volatility == 0:
        return 0.0
    return mean(daily_returns) / volatility * sqrt(TRADING_DAYS)


def annualized_return(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
    years = (len(equity_curve) - 1) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1.0


def max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def rsi(closes: list[float], period: int, index: int) -> float | None:
    if index < period:
        return None
    gains = 0.0
    losses = 0.0
    for previous, current in zip(closes[index - period : index], closes[index - period + 1 : index + 1]):
        change = current - previous
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    relative_strength = gains / losses
    return 100.0 - (100.0 / (1.0 + relative_strength))

