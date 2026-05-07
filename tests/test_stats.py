import math
import pytest
from quant_lab.tournament.stats import compute_metrics, Metrics


def test_compute_metrics_flat_curve():
    nav = [100_000] * 252
    m = compute_metrics(nav)
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0


def test_compute_metrics_steady_growth():
    nav = [100_000 * ((1.001) ** i) for i in range(252)]
    m = compute_metrics(nav)
    assert m.total_return > 0
    assert m.sharpe > 0
    assert m.max_drawdown == 0.0


def test_compute_metrics_drawdown():
    nav = [100, 110, 120, 90, 100]
    m = compute_metrics(nav)
    # Max DD from 120 to 90 = -25%
    assert math.isclose(m.max_drawdown, -0.25, abs_tol=1e-9)


def test_compute_metrics_single_point_safe():
    m = compute_metrics([100])
    assert m.total_return == 0.0
    assert m.sharpe == 0.0
