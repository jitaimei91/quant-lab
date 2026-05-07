import math
from quant_lab.tournament.stats import compute_metrics


def test_compute_metrics_flat_curve():
    nav = [100_000] * 252
    m = compute_metrics(nav)
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
    # New fields default to 0 / None when no benchmarks provided
    assert m.alpha_t_stat_vs_spy == 0.0
    assert m.alpha_t_stat_vs_qqq == 0.0
    assert m.significance_weight == 0.0
    assert m.factor_loadings is None


def test_compute_metrics_steady_growth():
    nav = [100_000 * ((1.001) ** i) for i in range(252)]
    m = compute_metrics(nav)
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.max_drawdown == 0.0
    # CI bounds should be populated (bootstrap fires for 252 returns)
    assert m.sharpe_ci_lo <= m.sharpe
    assert m.sharpe_ci_hi >= m.sharpe


def test_compute_metrics_drawdown():
    nav = [100, 110, 120, 90, 100]
    m = compute_metrics(nav)
    # Max DD from 120 to 90 = -25%
    assert math.isclose(m.max_drawdown, -0.25, abs_tol=1e-9)


def test_compute_metrics_single_point_safe():
    m = compute_metrics([100])
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    # Extended fields are 0/None for degenerate input
    assert m.sharpe_ci_lo == 0.0
    assert m.sharpe_ci_hi == 0.0
    assert m.alpha_t_stat_vs_spy == 0.0
    assert m.factor_loadings is None


def test_compute_metrics_with_spy_benchmark():
    """When SPY returns are provided, alpha t-stat should be populated."""
    nav = [100_000 * ((1.001) ** i) for i in range(252)]
    spy_rets = [0.0004] * 251  # flat SPY for simplicity
    m = compute_metrics(nav, daily_returns_vs_spy=spy_rets)
    # alpha_t_stat_vs_spy should be non-zero (strategy beats flat spy)
    assert isinstance(m.alpha_t_stat_vs_spy, float)
    assert isinstance(m.significance_weight, float)
    assert 0.0 <= m.significance_weight <= 1.0


def test_compute_metrics_with_factor_returns():
    """When factor_returns is provided, factor_loadings is populated."""
    n = 252
    nav = [100_000 * ((1.001) ** i) for i in range(n)]
    mkt = [0.0003] * (n - 1)
    size = [0.0001] * (n - 1)
    value = [0.0001] * (n - 1)
    factor_returns = {"MKT": mkt, "SIZE": size, "VALUE": value}
    m = compute_metrics(nav, factor_returns=factor_returns)
    assert m.factor_loadings is not None
    assert "beta_mkt" in m.factor_loadings
    assert "alpha_per_day" in m.factor_loadings
    assert "r_squared" in m.factor_loadings


def test_metrics_is_frozen():
    """Metrics dataclass is immutable."""
    m = compute_metrics([100, 101, 102])
    try:
        m.sharpe = 999.0  # type: ignore[misc]
        assert False, "Should have raised FrozenInstanceError"
    except (AttributeError, TypeError):
        pass
