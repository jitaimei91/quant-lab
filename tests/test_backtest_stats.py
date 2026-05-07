# tests/test_backtest_stats.py
import math
import random

from quant_lab.backtest.stats import (
    block_bootstrap_sharpe_ci,
    alpha_t_stat_vs_benchmark,
    significance_weight,
)


def _series(seed, mean=0.0005, vol=0.012, n=252):
    rng = random.Random(seed)
    return [rng.gauss(mean, vol) for _ in range(n)]


def test_block_bootstrap_sharpe_ci_returns_interval_around_point_estimate():
    rets = _series(0)
    point, lo, hi = block_bootstrap_sharpe_ci(rets, n_iter=500, block_len=20, seed=1)
    assert lo <= point <= hi
    # Width is positive
    assert hi > lo


def test_block_bootstrap_handles_short_series_safely():
    point, lo, hi = block_bootstrap_sharpe_ci([0.01, -0.005], n_iter=10, block_len=2, seed=1)
    assert math.isfinite(point) or point == 0.0


def test_alpha_t_stat_positive_when_strategy_beats_benchmark():
    strat = _series(0, mean=0.001)
    bench = _series(1, mean=0.0003)
    alpha, t = alpha_t_stat_vs_benchmark(strat, bench)
    assert alpha > 0
    assert t > 0


def test_significance_weight_zero_below_threshold():
    assert significance_weight(t_stat=0.5) == 0.0
    assert 0 < significance_weight(t_stat=1.5) < 1.0
    assert significance_weight(t_stat=3.0) == 1.0
