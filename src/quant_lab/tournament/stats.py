"""Performance statistics for paper-traded strategies.

Phase 1 ships total return, annualized return, Sharpe, max drawdown.
Phase 2 will add bootstrapped confidence intervals and Fama-French
factor decomposition (spec section 9).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Metrics:
    total_return: float
    annualized_return: float
    sharpe: float
    volatility: float
    max_drawdown: float
    days: int


def _daily_returns(nav: list[float]) -> list[float]:
    rets = []
    for i in range(1, len(nav)):
        prev = nav[i - 1]
        if prev <= 0:
            return []
        rets.append(nav[i] / prev - 1.0)
    return rets


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return sqrt(var)


def _max_drawdown(nav: list[float]) -> float:
    if not nav:
        return 0.0
    peak = nav[0]
    worst = 0.0
    for value in nav:
        peak = max(peak, value)
        if peak > 0:
            dd = (value - peak) / peak
            worst = min(worst, dd)
    return worst


def compute_metrics(nav: list[float]) -> Metrics:
    """Compute standard performance metrics from a NAV time series."""
    if len(nav) < 2:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, len(nav))

    rets = _daily_returns(nav)
    if not rets:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, len(nav))

    total = nav[-1] / nav[0] - 1.0
    daily_mean = sum(rets) / len(rets)
    daily_vol = _stdev(rets)
    ann_vol = daily_vol * sqrt(TRADING_DAYS_PER_YEAR)
    ann_return = (1 + daily_mean) ** TRADING_DAYS_PER_YEAR - 1
    sharpe = (daily_mean / daily_vol) * sqrt(TRADING_DAYS_PER_YEAR) if daily_vol > 0 else 0.0
    return Metrics(
        total_return=total,
        annualized_return=ann_return,
        sharpe=sharpe,
        volatility=ann_vol,
        max_drawdown=_max_drawdown(nav),
        days=len(nav),
    )
