"""Performance statistics for paper-traded strategies.

Phase 1 ships total return, annualized return, Sharpe, max drawdown.
Phase 2 adds bootstrapped confidence intervals, 3-factor decomposition,
and significance flags (spec section 9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    sharpe_ci_lo: float = 0.0
    sharpe_ci_hi: float = 0.0
    alpha_t_stat_vs_spy: float = 0.0
    alpha_t_stat_vs_qqq: float = 0.0
    significance_weight: float = 0.0
    factor_loadings: dict[str, float] | None = field(default=None, compare=True)


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


def compute_metrics(
    nav: list[float],
    *,
    daily_returns_vs_spy: list[float] | None = None,
    daily_returns_vs_qqq: list[float] | None = None,
    factor_returns: dict[str, list[float]] | None = None,
) -> Metrics:
    """Compute standard performance metrics from a NAV time series.

    Parameters
    ----------
    nav:
        Portfolio NAV series (absolute values, not returns).
    daily_returns_vs_spy:
        Aligned SPY daily-return series; when provided, alpha t-stat vs SPY
        and significance weight are computed.
    daily_returns_vs_qqq:
        Aligned QQQ daily-return series; when provided, alpha t-stat vs QQQ
        is computed.
    factor_returns:
        Dict with keys "MKT", "SIZE", "VALUE"; when provided, 3-factor OLS
        decomposition is added to factor_loadings.
    """
    if len(nav) < 2:
        return Metrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe=0.0,
            sharpe_ci_lo=0.0,
            sharpe_ci_hi=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            days=len(nav),
            alpha_t_stat_vs_spy=0.0,
            alpha_t_stat_vs_qqq=0.0,
            significance_weight=0.0,
            factor_loadings=None,
        )

    rets = _daily_returns(nav)
    if not rets:
        return Metrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe=0.0,
            sharpe_ci_lo=0.0,
            sharpe_ci_hi=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            days=len(nav),
            alpha_t_stat_vs_spy=0.0,
            alpha_t_stat_vs_qqq=0.0,
            significance_weight=0.0,
            factor_loadings=None,
        )

    total = nav[-1] / nav[0] - 1.0
    daily_mean = sum(rets) / len(rets)
    daily_vol = _stdev(rets)
    ann_vol = daily_vol * sqrt(TRADING_DAYS_PER_YEAR)
    ann_return = (1 + daily_mean) ** TRADING_DAYS_PER_YEAR - 1
    sharpe = (daily_mean / daily_vol) * sqrt(TRADING_DAYS_PER_YEAR) if daily_vol > 0 else 0.0

    # Bootstrap CI for Sharpe
    from ..backtest.stats import (
        block_bootstrap_sharpe_ci,
        alpha_t_stat_vs_benchmark,
        significance_weight as _sig_weight,
    )

    _, ci_lo, ci_hi = block_bootstrap_sharpe_ci(rets, seed=42)

    # Alpha t-stats
    alpha_t_spy = 0.0
    alpha_t_qqq = 0.0
    if daily_returns_vs_spy is not None:
        _, alpha_t_spy = alpha_t_stat_vs_benchmark(rets, daily_returns_vs_spy)
    if daily_returns_vs_qqq is not None:
        _, alpha_t_qqq = alpha_t_stat_vs_benchmark(rets, daily_returns_vs_qqq)

    sig_weight = _sig_weight(alpha_t_spy)

    # Factor loadings
    loadings: dict[str, float] | None = None
    if factor_returns is not None:
        from .factors import compute_factor_loadings
        loadings = compute_factor_loadings(rets, factor_returns)

    return Metrics(
        total_return=total,
        annualized_return=ann_return,
        sharpe=sharpe,
        sharpe_ci_lo=ci_lo,
        sharpe_ci_hi=ci_hi,
        volatility=ann_vol,
        max_drawdown=_max_drawdown(nav),
        days=len(nav),
        alpha_t_stat_vs_spy=alpha_t_spy,
        alpha_t_stat_vs_qqq=alpha_t_qqq,
        significance_weight=sig_weight,
        factor_loadings=loadings,
    )
