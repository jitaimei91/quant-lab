"""Tests for tournament/factors.py — 3-factor decomposition."""
from __future__ import annotations

import math
from datetime import date

import pytest

from quant_lab.tournament.factors import compute_factor_loadings, factor_proxies_from_histories
from quant_lab.types import Bar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str, d: date, close: float) -> Bar:
    return Bar(symbol=symbol, date=d, open=close, high=close, low=close, close=close, volume=1_000_000)


def _returns_from_nav(nav: list[float]) -> list[float]:
    return [nav[i] / nav[i - 1] - 1.0 for i in range(1, len(nav))]


def _make_histories(n: int = 200):
    """Return minimal histories dict with SPY, IWM, VTV, VUG bars."""
    import random
    rng = random.Random(0)
    result = {}
    for sym in ("SPY", "IWM", "VTV", "VUG"):
        price = 100.0
        bars = []
        for i in range(n):
            d = date(2024, 1, 2).fromordinal(date(2024, 1, 2).toordinal() + i)
            price *= 1.0 + rng.gauss(0.0003, 0.01)
            bars.append(_bar(sym, d, max(price, 1.0)))
        result[sym] = bars
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pure_mkt_beta():
    """Strategy = 1.5 * MKT + negligible SIZE/VALUE → beta_mkt ≈ 1.5, alpha ≈ 0."""
    import random
    rng = random.Random(42)
    n = 500
    mkt = [rng.gauss(0.0003, 0.01) for _ in range(n)]
    # Strategy is exactly 1.5 * MKT (no noise for a clean test)
    strategy = [1.5 * m for m in mkt]
    # Size and value must be non-degenerate for the OLS design matrix to be
    # full-rank. Use independent noise with realistic magnitude.
    size = [rng.gauss(0.0, 0.002) for _ in range(n)]
    value = [rng.gauss(0.0, 0.002) for _ in range(n)]

    result = compute_factor_loadings(strategy, {"MKT": mkt, "SIZE": size, "VALUE": value})

    assert math.isclose(result["beta_mkt"], 1.5, abs_tol=0.05)
    assert math.isclose(result["alpha_per_day"], 0.0, abs_tol=1e-4)
    assert math.isclose(result["beta_size"], 0.0, abs_tol=0.05)
    assert math.isclose(result["beta_value"], 0.0, abs_tol=0.05)
    assert result["r_squared"] > 0.95


def test_insufficient_data_returns_zeros():
    """Fewer than 30 data points → all zeros."""
    result = compute_factor_loadings(
        [0.01] * 10,
        {"MKT": [0.01] * 10, "SIZE": [0.0] * 10, "VALUE": [0.0] * 10},
    )
    assert result["beta_mkt"] == 0.0
    assert result["alpha_per_day"] == 0.0
    assert result["r_squared"] == 0.0


def test_factor_proxies_from_histories_keys():
    """factor_proxies_from_histories returns MKT, SIZE, VALUE keys."""
    histories = _make_histories(100)
    proxies = factor_proxies_from_histories(histories)

    assert set(proxies.keys()) == {"MKT", "SIZE", "VALUE"}
    assert len(proxies["MKT"]) > 0
    # All series should be aligned (same length)
    assert len(proxies["MKT"]) == len(proxies["SIZE"]) == len(proxies["VALUE"])


def test_factor_proxies_missing_symbol_returns_empty():
    """Missing any required symbol → empty dict."""
    histories = _make_histories(100)
    del histories["IWM"]  # remove one required proxy
    proxies = factor_proxies_from_histories(histories)
    assert proxies == {}


def test_multi_factor_decomposition():
    """Strategy driven by MKT + SIZE → both betas recovered correctly."""
    import random
    rng = random.Random(7)
    n = 500
    mkt = [rng.gauss(0.0003, 0.01) for _ in range(n)]
    size = [rng.gauss(0.0, 0.005) for _ in range(n)]
    value = [rng.gauss(0.0, 0.005) for _ in range(n)]
    strategy = [1.2 * mkt[i] + 0.8 * size[i] for i in range(n)]

    result = compute_factor_loadings(strategy, {"MKT": mkt, "SIZE": size, "VALUE": value})

    assert math.isclose(result["beta_mkt"], 1.2, abs_tol=0.02)
    assert math.isclose(result["beta_size"], 0.8, abs_tol=0.02)
    assert math.isclose(result["alpha_per_day"], 0.0, abs_tol=1e-5)
    assert result["r_squared"] > 0.95
