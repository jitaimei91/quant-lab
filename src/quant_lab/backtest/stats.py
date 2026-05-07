# src/quant_lab/backtest/stats.py
"""Statistical utilities for backtest calibration.

Block bootstrap respects time-series autocorrelation (daily returns are not iid).
Alpha t-stat is from the regression of strategy returns on benchmark returns;
the significance weight maps t-stat to a [0, 1] confidence factor.
"""
from __future__ import annotations

import random
import statistics
from math import sqrt


TRADING_DAYS_PER_YEAR = 252


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = statistics.mean(returns)
    std = statistics.stdev(returns)
    if std == 0:
        return 0.0
    return (mean / std) * sqrt(TRADING_DAYS_PER_YEAR)


def block_bootstrap_sharpe_ci(
    returns: list[float],
    n_iter: int = 1000,
    block_len: int = 20,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Stationary block bootstrap CI for Sharpe ratio.

    Returns (point_estimate, lo_2_5, hi_97_5).
    """
    if len(returns) < 2:
        return 0.0, 0.0, 0.0
    point = _sharpe(returns)
    if len(returns) < block_len:
        return point, point, point

    rng = random.Random(seed)
    n = len(returns)
    samples: list[float] = []
    for _ in range(n_iter):
        resampled: list[float] = []
        while len(resampled) < n:
            start = rng.randrange(0, n - block_len + 1)
            resampled.extend(returns[start : start + block_len])
        samples.append(_sharpe(resampled[:n]))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return point, lo, hi


def alpha_t_stat_vs_benchmark(
    strategy_returns: list[float],
    benchmark_returns: list[float],
) -> tuple[float, float]:
    """Regress strategy on benchmark; return (alpha_per_day, t_stat_of_alpha)."""
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 30:
        return 0.0, 0.0
    s = strategy_returns[:n]
    b = benchmark_returns[:n]
    s_mean = statistics.mean(s)
    b_mean = statistics.mean(b)
    cov = sum((s[i] - s_mean) * (b[i] - b_mean) for i in range(n)) / (n - 1)
    b_var = sum((b[i] - b_mean) ** 2 for i in range(n)) / (n - 1)
    if b_var == 0:
        return 0.0, 0.0
    beta = cov / b_var
    alpha = s_mean - beta * b_mean
    residuals = [s[i] - (alpha + beta * b[i]) for i in range(n)]
    res_var = sum(r * r for r in residuals) / (n - 2) if n > 2 else 0
    if res_var <= 0:
        return alpha, 0.0
    se_alpha = sqrt(res_var * (1.0 / n + b_mean * b_mean / ((n - 1) * b_var)))
    if se_alpha == 0:
        return alpha, 0.0
    return alpha, alpha / se_alpha


def significance_weight(t_stat: float) -> float:
    """Map t-stat to [0, 1]. 0 below t=1, ramps linearly to 1 at t=3."""
    if t_stat <= 1.0:
        return 0.0
    if t_stat >= 3.0:
        return 1.0
    return (t_stat - 1.0) / 2.0
